"""
Multi-database sampler dataset for distributed training.

Supports multiple preprocessed database directories, distributing them
across workers with deterministic shuffling per epoch.

Design for fork-safety and DataLoader workers:
- Samplers are created lazily per-worker (not in __init__)
- Each worker gets its own Rust Sampler with its own thread pool
- Sharding accounts for both DDP rank and DataLoader worker_id
"""

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

import tributary
from dbtransformer.configurations import (
    DataConfig,
    DDPParameters,
    ModelConfig,
    SamplerConfig,
    TrainingConfig,
)
from dbtransformer.model import Batch


def discover_databases(db_dir: Path, db_names: list[str] | None = None) -> list[Path]:
    """
    Discover database directories in the given base directory.

    Args:
        db_dir: Base directory containing database subdirectories
        db_names: Optional list of specific database names to load.
                  If None, discovers all valid database subdirectories.

    Returns:
        Sorted list of paths to database directories.
    """
    if not db_dir.exists():
        raise ValueError(f"Database directory does not exist: {db_dir}")

    if db_names is not None:
        # Load specific databases by name
        db_paths = []
        for name in db_names:
            db_path = db_dir / name
            if not db_path.exists():
                raise ValueError(f"Database '{name}' not found in {db_dir}")
            if not (db_path / "schema.rkyv").exists():
                raise ValueError(f"Database '{name}' is missing schema.rkyv - was it preprocessed?")
            db_paths.append(db_path)
        return sorted(db_paths)
    # Discover all database subdirectories (those containing schema.rkyv)
    db_paths = []
    for subdir in db_dir.iterdir():
        if subdir.is_dir() and (subdir / "schema.rkyv").exists():
            db_paths.append(subdir)

    if not db_paths:
        raise ValueError(
            f"No preprocessed databases found in {db_dir}. Each database should be a subdirectory containing schema.rkyv, graph.rkyv, etc."
        )
    return sorted(db_paths)


class SamplerBatchDataset(IterableDataset[Batch]):
    """
    Multi-database sampler dataset that yields full batches.

    Fork-safe design for PyTorch DataLoader workers:
    - Stores only configuration in __init__ (no Sampler objects)
    - Creates Samplers lazily per-worker in __iter__
    - Each worker has its own thread pool, avoiding fork issues

    Sharding for distributed + multi-worker training:
    - DDP provides (rank, world_size) for GPU-level sharding
    - DataLoader provides (worker_id, num_workers) for CPU-level sharding
    - Effective parallelism = world_size * num_workers
    - Each (rank, worker_id) pair gets a disjoint subset of batches
    """

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        sampler_config: SamplerConfig,
        ddp_parameters: DDPParameters | None = None,
        precreate_samplers: bool = False,
    ) -> None:
        # Store configuration only - samplers created lazily per worker
        # Unless precreate_samplers=True, in which case we create them now
        # (safe for num_workers=0 cases like eval)
        self.batch_size = training_config.batch_size
        self.seq_len = training_config.seq_len
        self.d_text = model_config.d_text
        self.d_time = model_config.d_time
        self.sampler_config = sampler_config

        # DDP parameters for GPU-level sharding
        self._ddp_rank = ddp_parameters.global_rank if ddp_parameters else 0
        self._ddp_world_size = ddp_parameters.world_size if ddp_parameters else 1

        # Discover database directories (this is fast, just path checking)
        db_dir = Path(data_config.db_dir)
        self._db_paths = discover_databases(db_dir, data_config.db_names)

        # Pre-compute batch counts per database (quick metadata read)
        # We need this for __len__ and global index building
        self._db_batch_counts: list[int] = []
        self._total_batches = 0
        self._embed_dim: int | None = None

        for db_path in self._db_paths:
            # Create a temporary sampler just to get metadata
            # This is quick and we immediately drop it
            temp_sampler = tributary.Sampler(
                db_path=str(db_path),
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                max_bfs_width=sampler_config.max_bfs_width,
                seed=sampler_config.seed,
                num_threads=1,  # Minimal threads for metadata read
                split=sampler_config.split,
                train_frac=sampler_config.train_frac,
                val_frac=sampler_config.val_frac,
                split_seed=sampler_config.split_seed,
            )

            num_batches = temp_sampler.len_py()
            self._db_batch_counts.append(num_batches)
            self._total_batches += num_batches

            # Validate/store embedding dimension
            db_embed_dim = temp_sampler.embed_dim()
            if self._embed_dim is None:
                self._embed_dim = db_embed_dim
            elif self._embed_dim != db_embed_dim:
                raise ValueError(f"Inconsistent embedding dimensions: {db_path} has {db_embed_dim}, expected {self._embed_dim}")

            # Log info from rank 0
            if self._ddp_rank == 0 and sampler_config.split:
                num_seeds = temp_sampler.num_seeds()
                num_rows = temp_sampler.num_rows()
                print(f"  {db_path.name}: {num_seeds:,} seeds / {num_rows:,} rows ({sampler_config.split} split)")

        # Validate model's d_text matches database embedding dimension
        if self._embed_dim is not None and self.d_text != self._embed_dim:
            raise ValueError(
                f"Model d_text ({self.d_text}) doesn't match database embedding dim ({self._embed_dim}). Update ModelConfig.d_text to match."
            )

        # Build global batch index: list of (db_idx, batch_idx)
        self._global_indices: list[tuple[int, int]] = []
        for db_idx, num_batches in enumerate(self._db_batch_counts):
            for batch_idx in range(num_batches):
                self._global_indices.append((db_idx, batch_idx))

        self._epoch = 0

        # Pre-create samplers if requested (for num_workers=0 cases like eval)
        # This avoids recreating samplers on every __iter__ call
        self._cached_samplers: list[tributary.Sampler] | None = None
        if precreate_samplers:
            self._cached_samplers = []
            for db_path in self._db_paths:
                sampler = tributary.Sampler(
                    db_path=str(db_path),
                    batch_size=self.batch_size,
                    seq_len=self.seq_len,
                    max_bfs_width=sampler_config.max_bfs_width,
                    seed=sampler_config.seed,
                    num_threads=sampler_config.num_threads,
                    split=sampler_config.split,
                    train_frac=sampler_config.train_frac,
                    val_frac=sampler_config.val_frac,
                    split_seed=sampler_config.split_seed,
                )
                self._cached_samplers.append(sampler)

        # Log database info
        if self._ddp_rank == 0:
            print(f"Loaded {len(self._db_paths)} databases, {self._total_batches:,} total batches")

    def __len__(self) -> int:
        """Number of batches this DDP rank will process per epoch (across all workers)."""
        # Note: This doesn't account for num_workers since that's set on DataLoader
        return (self._total_batches + self._ddp_world_size - 1 - self._ddp_rank) // self._ddp_world_size

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling."""
        self._epoch = epoch
        # Note: Actual shuffling happens in __iter__ since we create samplers there

    def __iter__(self):
        """
        Yield batches for this worker.

        Creates fresh Samplers per-worker to ensure fork-safety.
        Each (DDP rank, DataLoader worker) pair gets a disjoint subset.
        """
        # Get worker info for DataLoader-level sharding
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process data loading (num_workers=0)
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Compute effective rank across all parallel workers
        # Total parallelism = ddp_world_size * num_workers
        effective_world_size = self._ddp_world_size * num_workers
        effective_rank = self._ddp_rank * num_workers + worker_id

        # Use cached samplers if available, otherwise create fresh ones
        # (fresh creation is needed for fork-safety with num_workers > 0)
        if self._cached_samplers is not None:
            samplers = self._cached_samplers
        else:
            samplers = []
            for db_path in self._db_paths:
                sampler = tributary.Sampler(
                    db_path=str(db_path),
                    batch_size=self.batch_size,
                    seq_len=self.seq_len,
                    max_bfs_width=self.sampler_config.max_bfs_width,
                    seed=self.sampler_config.seed,
                    num_threads=self.sampler_config.num_threads,
                    split=self.sampler_config.split,
                    train_frac=self.sampler_config.train_frac,
                    val_frac=self.sampler_config.val_frac,
                    split_seed=self.sampler_config.split_seed,
                )
                samplers.append(sampler)

        # Shuffle each sampler's internal seeds for the epoch
        for sampler in samplers:
            sampler.shuffle_py(self._epoch)

        # Shuffle global indices deterministically for this epoch
        shuffled_indices = self._global_indices.copy()
        rng = random.Random(self._epoch)
        rng.shuffle(shuffled_indices)

        # Stride through indices: each effective_rank gets every effective_world_size-th batch
        for global_idx in range(effective_rank, self._total_batches, effective_world_size):
            db_idx, batch_idx = shuffled_indices[global_idx]
            sampler = samplers[db_idx]

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
