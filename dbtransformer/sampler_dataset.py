from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import tributary
from torch.utils.data import Dataset

from dbtransformer.configurations import ModelConfig, SamplerConfig, TrainingConfig
from dbtransformer.model import Batch


class SamplerBatchDataset(Dataset[Batch]):
    """
    Wrap the Rust Sampler (PyO3) as a PyTorch Dataset that yields full batches.

    Each __getitem__ returns a pre-batched `Batch`, so DataLoader should use
    batch_size=None and no collate_fn.

    The sampler loads a single preprocessed database (.rkyv file) and samples
    BFS neighborhoods around seed rows, with configurable masking.
    """

    def __init__(
        self,
        db_path: str | Path,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        sampler_config: SamplerConfig,
    ) -> None:
        self.batch_size = training_config.batch_size
        self.seq_len = training_config.seq_len
        self.d_text = model_config.d_text
        self.d_time = model_config.d_time

        self._sampler = tributary.Sampler(
            db_path=str(db_path),
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            max_bfs_width=sampler_config.max_bfs_width,
            mask_rate=0.15,  # TODO: make configurable via SamplerConfig
            seed=sampler_config.seed,
        )
        self._num_batches = self._sampler.len_py()

    def __len__(self) -> int:
        return self._num_batches

    def __getitem__(self, idx: int) -> Batch:
        # Rust returns a list of (name, numpy array) pairs
        raw = dict(self._sampler.batch_py(idx))

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

    def set_epoch(self, epoch: int) -> None:
        """Update sampler epoch and shuffle order inside Rust."""
        self._sampler.shuffle_py(epoch)
