from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

import tributary
from dbtransformer.configurations import DDPParameters, ModelConfig, TrainingConfig
from dbtransformer.model import Batch


class SamplerBatchDataset(Dataset[Batch]):
    """
    Wrap the Rust Sampler (PyO3) as a PyTorch Dataset that yields full batches.

    Each __getitem__ returns a pre-batched `Batch`, so DataLoader should use
    batch_size=None and no collate_fn. The Rust sampler internally partitions
    batches by rank/world_size, so we do NOT wrap this dataset with a
    DistributedSampler.
    """

    def __init__(
        self,
        sampler_config: SamplerConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        ddp_parameters: DDPParameters,
    ) -> None:
        self.batch_size = training_config.batch_size
        self.seq_len = training_config.seq_len
        self.d_text = model_config.d_text

        self._sampler = tributary.Sampler(
            db_configs=db_configs,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            rank=ddp_parameters.global_rank,
            world_size=ddp_parameters.world_size,
            max_bfs_width=data_config.max_bfs_width,
            d_text=self.d_text,
            seed=data_config.seed,
        )
        self._num_batches = self._sampler.len_py()

    def __len__(self) -> int:
        return self._num_batches

    def __getitem__(self, idx: int) -> Batch:
        # Rust returns a list of (name, numpy array/value) pairs
        raw = dict(self._sampler.batch_py(idx))

        def to_tensor(name: str, dtype: torch.dtype | None = None, shape: Sequence[int] | None = None) -> torch.Tensor:
            arr = raw[name]
            tensor = torch.from_numpy(np.array(arr, copy=False))
            if dtype is not None:
                tensor = tensor.to(dtype)
            if shape is not None:
                tensor = tensor.view(*shape)
            return tensor

        number_values = to_tensor("number_values", torch.float32, (self.batch_size, self.seq_len, 1))
        datetime_values = to_tensor("datetime_values", torch.float32, (self.batch_size, self.seq_len, 1))
        boolean_values = to_tensor("boolean_values", torch.float32, (self.batch_size, self.seq_len, 1))
        text_values = to_tensor("text_values", torch.float32, (self.batch_size, self.seq_len, self.d_text))
        column_name_values = to_tensor("column_name_values", torch.float32, (self.batch_size, self.seq_len, self.d_text))
        semantic_types = to_tensor("semantic_types", torch.long, (self.batch_size, self.seq_len))
        masks = to_tensor("masks", torch.bool, (self.batch_size, self.seq_len))
        is_task_node = to_tensor("is_task_node", torch.bool, (self.batch_size, self.seq_len))
        is_padding = to_tensor("is_padding", torch.bool, (self.batch_size, self.seq_len))

        column_attn_mask = to_tensor("column_attn_mask", torch.bool, (self.batch_size, self.seq_len, self.seq_len))
        feature_attn_mask = to_tensor("feature_attn_mask", torch.bool, (self.batch_size, self.seq_len, self.seq_len))
        neighbor_attn_mask = to_tensor("neighbor_attn_mask", torch.bool, (self.batch_size, self.seq_len, self.seq_len))

        return Batch(
            number_values=number_values,
            datetime_values=datetime_values,
            boolean_values=boolean_values,
            text_values=text_values,
            column_name_values=column_name_values,
            semantic_types=semantic_types,
            masks=masks,
            is_task_node=is_task_node,
            is_padding=is_padding,
            column_attn_mask=column_attn_mask,
            feature_attn_mask=feature_attn_mask,
            neighbor_attn_mask=neighbor_attn_mask,
        )

    def set_epoch(self, epoch: int) -> None:
        """Update sampler epoch and shuffle order inside Rust."""
        self._sampler.shuffle_py(epoch)
