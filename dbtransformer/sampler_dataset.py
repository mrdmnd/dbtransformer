"""
Multi-database sampler dataset for distributed training.

Supports multiple .rkyv database files in a directory, distributing them
across workers with deterministic shuffling per epoch.
"""

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import tributary
from torch.utils.data import IterableDataset

from dbtransformer.configurations import (
    DataConfig,
    DDPParameters,
    ModelConfig,
    SamplerConfig,
    TrainingConfig,
)
from dbtransformer.model import Batch


class SamplerBatchDataset(IterableDataset[Batch]):
    """
    Multi-database sampler dataset that yields full batches.

    Scans a directory for .rkyv files, creates a Sampler for each, and
    yields batches in a deterministic order that's shuffled per epoch.

    For distributed training, each rank works on a different subset of
    (database, batch) pairs via deterministic sharding.

    Key design decisions:
    - Each batch comes from a single database (keeps BFS traversal coherent)
    - Databases are weighted by row count for balanced sampling
    - Shuffling is deterministic given epoch, for reproducibility
    - Workers shard the global batch sequence by rank
    """

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        sampler_config: SamplerConfig,
        ddp_parameters: DDPParameters | None = None,
    ) -> None:
        self.batch_size = training_config.batch_size
        self.seq_len = training_config.seq_len
        self.d_text = model_config.d_text
        self.d_time = model_config.d_time
        self.sampler_config = sampler_config

        # DDP parameters for distributing batches across ranks
        self._rank = ddp_parameters.global_rank if ddp_parameters else 0
        self._world_size = ddp_parameters.world_size if ddp_parameters else 1

        # Discover all .rkyv files in the directory
        db_dir = Path(data_config.db_dir)
        if not db_dir.exists():
            raise ValueError(f"Database directory does not exist: {db_dir}")

        self._db_paths = sorted(db_dir.glob("*.rkyv"))
        if not self._db_paths:
            raise ValueError(f"No .rkyv files found in {db_dir}")

        # Create samplers for each database
        # Store as list of (path, sampler, num_batches)
        self._samplers: list[tuple[Path, tributary.Sampler, int]] = []
        total_embed_dim = None

        for db_path in self._db_paths:
            sampler = tributary.Sampler(
                db_path=str(db_path),
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                max_bfs_width=sampler_config.max_bfs_width,
                seed=sampler_config.seed,
                num_threads=sampler_config.num_threads,
            )

            # Validate embedding dimensions are consistent across DBs
            db_embed_dim = sampler.embed_dim()
            if total_embed_dim is None:
                total_embed_dim = db_embed_dim
            elif total_embed_dim != db_embed_dim:
                raise ValueError(
                    f"Inconsistent embedding dimensions: {db_path} has {db_embed_dim}, "
                    f"expected {total_embed_dim}"
                )

            num_batches = sampler.len_py()
            self._samplers.append((db_path, sampler, num_batches))

        # Validate model's d_text matches database embedding dimension
        if total_embed_dim is not None and self.d_text != total_embed_dim:
            raise ValueError(
                f"Model d_text ({self.d_text}) doesn't match database embedding dim "
                f"({total_embed_dim}). Update ModelConfig.d_text to match."
            )

        # Build global batch index: list of (db_idx, batch_idx) covering all batches
        self._global_indices: list[tuple[int, int]] = []
        for db_idx, (_, _, num_batches) in enumerate(self._samplers):
            for batch_idx in range(num_batches):
                self._global_indices.append((db_idx, batch_idx))

        self._total_batches = len(self._global_indices)
        self._epoch = 0

        # Log database info
        if self._rank == 0:
            total_rows = sum(s.num_rows() for _, s, _ in self._samplers)
            print(
                f"Loaded {len(self._samplers)} databases with {total_rows:,} total rows, "
                f"{self._total_batches:,} total batches"
            )

    def __len__(self) -> int:
        """Number of batches this worker will process per epoch."""
        return (self._total_batches + self._world_size - 1 - self._rank) // self._world_size

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for deterministic shuffling.

        This shuffles the global batch indices and each sampler's internal seeds.
        All workers use the same epoch, ensuring they see disjoint subsets.
        """
        self._epoch = epoch

        # Shuffle each sampler's internal seed order
        for _, sampler, _ in self._samplers:
            sampler.shuffle_py(epoch)

        # Shuffle global batch indices deterministically
        rng = random.Random(epoch)
        rng.shuffle(self._global_indices)

    def __iter__(self):
        """
        Yield batches for this worker.

        Iterates through the shuffled global indices, skipping to this worker's
        assigned batches via striding.
        """
        # Stride through global indices: rank 0 gets 0, W, 2W, ...; rank 1 gets 1, W+1, ...
        for global_idx in range(self._rank, self._total_batches, self._world_size):
            db_idx, batch_idx = self._global_indices[global_idx]
            _, sampler, _ = self._samplers[db_idx]

            # Get raw batch from Rust sampler
            raw = dict(sampler.batch_py(batch_idx))
            yield self._raw_to_batch(raw)

    def _raw_to_batch(self, raw: dict) -> Batch:
        """Convert raw sampler output to Batch object."""

        def to_tensor(
            name: str,
            dtype: torch.dtype | None = None,
            shape: Sequence[int] | None = None,
        ) -> torch.Tensor:
            arr = raw[name]
            tensor = torch.from_numpy(np.array(arr, copy=False))
            if dtype is not None:
                tensor = tensor.to(dtype)
            if shape is not None:
                tensor = tensor.view(*shape)
            return tensor

        # Cell values by semantic type
        numerical_values = to_tensor(
            "numerical_values",
            torch.float32,
            (self.batch_size, self.seq_len, 1),
        )
        categorical_values = to_tensor(
            "categorical_values",
            torch.float32,
            (self.batch_size, self.seq_len, self.d_text),
        )
        text_values = to_tensor(
            "text_values",
            torch.float32,
            (self.batch_size, self.seq_len, self.d_text),
        )
        timestamp_values = to_tensor(
            "timestamp_values",
            torch.float32,
            (self.batch_size, self.seq_len, self.d_time),
        )
        column_name_values = to_tensor(
            "column_name_values",
            torch.float32,
            (self.batch_size, self.seq_len, self.d_text),
        )

        # Metadata
        semantic_types = to_tensor(
            "semantic_types",
            torch.long,
            (self.batch_size, self.seq_len),
        )
        masks = to_tensor(
            "masks",
            torch.bool,
            (self.batch_size, self.seq_len),
        )
        is_padding = to_tensor(
            "is_padding",
            torch.bool,
            (self.batch_size, self.seq_len),
        )

        # Attention masks
        column_attn_mask = to_tensor(
            "column_attn_mask",
            torch.bool,
            (self.batch_size, self.seq_len, self.seq_len),
        )
        feature_attn_mask = to_tensor(
            "feature_attn_mask",
            torch.bool,
            (self.batch_size, self.seq_len, self.seq_len),
        )
        neighbor_attn_mask = to_tensor(
            "neighbor_attn_mask",
            torch.bool,
            (self.batch_size, self.seq_len, self.seq_len),
        )

        return Batch(
            numerical_values=numerical_values,
            categorical_values=categorical_values,
            text_values=text_values,
            timestamp_values=timestamp_values,
            column_name_values=column_name_values,
            semantic_types=semantic_types,
            masks=masks,
            is_padding=is_padding,
            column_attn_mask=column_attn_mask,
            feature_attn_mask=feature_attn_mask,
            neighbor_attn_mask=neighbor_attn_mask,
        )
