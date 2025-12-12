# This file contains code for generating "dummy data" for the model, just for
# initial testing and development.

from typing import TypedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from dbtransformer.configurations import DummyDataConfig, ModelConfig, TrainingConfig
from dbtransformer.model import MAX_F2P_NEIGHBORS, Batch
from dbtransformer.sampler_types import SemanticType


class DummySample(TypedDict):
    """A single sample (no batch dimension) for dummy data generation."""

    node_indices: Tensor  # (seq_len,)
    table_name_indices: Tensor  # (seq_len,)
    column_name_indices: Tensor  # (seq_len,)
    f2p_neighbor_indices: Tensor  # (seq_len, MAX_F2P_NEIGHBORS)
    number_values: Tensor  # (seq_len, 1)
    datetime_values: Tensor  # (seq_len, 1)
    boolean_values: Tensor  # (seq_len, 1)
    text_values: Tensor  # (seq_len, d_text)
    column_name_values: Tensor  # (seq_len, d_text)
    semantic_types: Tensor  # (seq_len,)
    masks: Tensor  # (seq_len,)
    is_targets: Tensor  # (seq_len,)
    is_task_nodes: Tensor  # (seq_len,)
    is_padding: Tensor  # (seq_len,)
    class_value_indices: Tensor  # (seq_len,)


def create_dummy_sample(seq_len: int, d_text: int) -> DummySample:
    """
    Create a single dummy sample with random data for testing.

    All tensors have shape (seq_len, ...) with NO batch dimension.
    The DataLoader will handle batching via the collate function.
    """
    # Create node indices: groups of cells belong to the same row
    # We'll simulate ~10 cells per row on average
    cells_per_row = 10
    num_rows = (seq_len + cells_per_row - 1) // cells_per_row
    node_indices = torch.arange(num_rows)
    node_indices = node_indices.repeat_interleave(cells_per_row)[:seq_len]
    node_indices = node_indices.to(torch.int32)

    # Table and column indices: simulate ~5 tables, ~20 columns per table
    num_tables = 5
    num_cols_per_table = 20
    table_indices = torch.randint(0, num_tables, (seq_len,), dtype=torch.int32)
    column_indices = torch.randint(0, num_tables * num_cols_per_table, (seq_len,), dtype=torch.int32)

    # Foreign-to-primary neighbor indices
    # Each cell has up to MAX_F2P_NEIGHBORS parent row references
    # Use -1 for padding (no neighbor)
    f2p = torch.randint(-1, num_rows, (seq_len, MAX_F2P_NEIGHBORS), dtype=torch.int32)

    # Numeric values (z-score normalized)
    # Use float32 here; converted to bfloat16 on CUDA via Batch.to_device()
    number_vals = torch.randn(seq_len, 1, dtype=torch.float32)
    datetime_vals = torch.randn(seq_len, 1, dtype=torch.float32)
    boolean_vals = torch.randn(seq_len, 1, dtype=torch.float32)

    # Text embeddings (pre-computed sentence transformer output)
    text_vals = torch.randn(seq_len, d_text, dtype=torch.float32)
    column_name_vals = torch.randn(seq_len, d_text, dtype=torch.float32)

    # Semantic types: 0=number, 1=text, 2=datetime, 3=boolean
    semantic_types = torch.randint(0, 4, (seq_len,), dtype=torch.long)

    # Masks: positions to hide and predict
    # Mask ~15% of non-text positions (model doesn't support text masking)
    masks = torch.rand(seq_len) < 0.15  # noqa: PLR2004
    masks &= semantic_types != SemanticType.TEXT.value

    # Ensure at least one masked position for loss computation
    if not masks.any():
        non_text = (semantic_types != SemanticType.TEXT.value).nonzero(as_tuple=True)[0]
        if len(non_text) > 0:
            masks[non_text[0]] = True

    # is_targets: same as masks for now
    is_targets = masks.clone()

    # is_task_nodes: first ~10% of rows are task table rows
    task_row_threshold = num_rows // 10
    is_task_nodes = node_indices < task_row_threshold

    # No padding in dummy data
    is_padding = torch.zeros(seq_len, dtype=torch.bool)

    # Class value indices (unused, set to -1)
    class_indices = torch.full((seq_len,), -1, dtype=torch.int32)

    return DummySample(
        node_indices=node_indices,
        table_name_indices=table_indices,
        column_name_indices=column_indices,
        f2p_neighbor_indices=f2p,
        number_values=number_vals,
        datetime_values=datetime_vals,
        boolean_values=boolean_vals,
        text_values=text_vals,
        column_name_values=column_name_vals,
        semantic_types=semantic_types,
        masks=masks,
        is_targets=is_targets,
        is_task_nodes=is_task_nodes,
        is_padding=is_padding,
        class_value_indices=class_indices,
    )


def create_dummy_batch(batch_size: int, seq_len: int, d_text: int) -> Batch:
    """Create a full pre-batched Batch directly (no collation needed).

    Args:
        batch_size: Number of samples in the batch.
        seq_len: Sequence length per sample.
        d_text: Dimension of text embeddings.
    """
    # Node indices: groups of cells belong to the same row
    cells_per_row = 10
    num_rows = (seq_len + cells_per_row - 1) // cells_per_row
    node_indices = torch.arange(num_rows).repeat_interleave(cells_per_row)
    node_indices = node_indices[:seq_len].unsqueeze(0).expand(batch_size, -1)
    node_indices = node_indices.to(torch.int32)

    # Table and column indices
    num_tables = 5
    num_cols_per_table = 20
    table_indices = torch.randint(0, num_tables, (batch_size, seq_len), dtype=torch.int32)
    column_indices = torch.randint(0, num_tables * num_cols_per_table, (batch_size, seq_len), dtype=torch.int32)

    # Foreign-to-primary neighbor indices
    f2p = torch.randint(-1, num_rows, (batch_size, seq_len, MAX_F2P_NEIGHBORS), dtype=torch.int32)

    # Numeric values (z-score normalized)
    number_vals = torch.randn(batch_size, seq_len, 1, dtype=torch.float32)
    datetime_vals = torch.randn(batch_size, seq_len, 1, dtype=torch.float32)
    boolean_vals = torch.randn(batch_size, seq_len, 1, dtype=torch.float32)

    # Text embeddings
    text_vals = torch.randn(batch_size, seq_len, d_text, dtype=torch.float32)
    column_name_vals = torch.randn(batch_size, seq_len, d_text, dtype=torch.float32)

    # Semantic types: 0=number, 1=text, 2=datetime, 3=boolean
    semantic_types = torch.randint(0, 4, (batch_size, seq_len), dtype=torch.long)

    # Masks: ~15% of non-text positions
    masks = torch.rand(batch_size, seq_len) < 0.15  # noqa: PLR2004
    masks &= semantic_types != SemanticType.TEXT.value
    # Ensure at least one masked position per sample
    masks[:, 0] = True
    masks[:, 0] &= semantic_types[:, 0] != SemanticType.TEXT.value

    # Task nodes: first ~10% of rows
    task_row_threshold = num_rows // 10
    is_task_node = node_indices < task_row_threshold

    # Padding: none for now.
    is_padding = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    # Compute dense boolean attention masks from the index tensors
    col_mask, feat_mask, neighbor_mask = Batch.compute_attention_masks(
        node_indices=node_indices,
        table_name_indices=table_indices,
        column_name_indices=column_indices,
        f2p_neighbor_indices=f2p,
        is_padding=is_padding,
    )

    return Batch(
        number_values=number_vals.contiguous(),
        datetime_values=datetime_vals.contiguous(),
        boolean_values=boolean_vals.contiguous(),
        text_values=text_vals.contiguous(),
        column_name_values=column_name_vals.contiguous(),
        semantic_types=semantic_types.contiguous(),
        masks=masks.contiguous(),
        is_task_node=is_task_node.contiguous(),
        is_padding=is_padding.contiguous(),
        column_attn_mask=col_mask.contiguous(),
        feature_attn_mask=feat_mask.contiguous(),
        neighbor_attn_mask=neighbor_mask.contiguous(),
    )


class PreBatchedDummyDataset(Dataset[Batch]):
    """
    Dataset that returns pre-batched Batches directly.

    No collation needed - DataLoader just passes through the Batch objects.
    This is MUCH faster than collating individual samples.
    """

    def __init__(
        self,
        data_config: DummyDataConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ) -> None:
        self.batch_size = training_config.batch_size
        self.seq_len = training_config.seq_len
        self.d_text = model_config.d_text

        # Number of batches (not samples!)
        self.num_batches = data_config.total_num_samples // self.batch_size

        # Pre-generate a pool of batches
        self._pool_size = min(64, self.num_batches)
        self._batch_pool: list[Batch] = [create_dummy_batch(self.batch_size, self.seq_len, self.d_text) for _ in range(self._pool_size)]

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, idx: int) -> Batch:
        return self._batch_pool[idx % self._pool_size]


# Keep old classes for backwards compatibility
class DummySampleDataset(Dataset[DummySample]):
    """
    Dataset that returns individual dummy samples for DataLoader batching.

    Pre-generates a pool of samples at initialization to avoid slow on-the-fly
    random tensor generation during training. Each __getitem__ returns a single
    sample (no batch dimension); the DataLoader handles batching.

    NOTE: Consider using PreBatchedDummyDataset instead for much faster loading.
    """

    def __init__(
        self,
        data_config: DummyDataConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ) -> None:
        self.data_config = data_config
        self.seq_len = training_config.seq_len
        self.d_text = model_config.d_text
        self.num_samples = data_config.total_num_samples

        # Pre-generate a pool of samples to cycle through
        self._pool_size = min(1000, self.num_samples)
        self._sample_pool: list[DummySample] = [create_dummy_sample(self.seq_len, self.d_text) for _ in range(self._pool_size)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> DummySample:
        return self._sample_pool[idx % self._pool_size]


def collate_samples(samples: list[DummySample]) -> Batch:
    """Collate a list of DummySample dicts into a single Batch.

    NOTE: This is slow! Consider using PreBatchedDummyDataset instead.
    """
    # Stack all tensors from samples
    node_indices = torch.stack([s["node_indices"] for s in samples])
    table_name_indices = torch.stack([s["table_name_indices"] for s in samples])
    column_name_indices = torch.stack([s["column_name_indices"] for s in samples])
    f2p_neighbor_indices = torch.stack([s["f2p_neighbor_indices"] for s in samples])
    is_padding = torch.stack([s["is_padding"] for s in samples])

    # Compute dense boolean attention masks from the index tensors
    col_mask, feat_mask, neighbor_mask = Batch.compute_attention_masks(
        node_indices=node_indices,
        table_name_indices=table_name_indices,
        column_name_indices=column_name_indices,
        f2p_neighbor_indices=f2p_neighbor_indices,
        is_padding=is_padding,
    )

    return Batch(
        number_values=torch.stack([s["number_values"] for s in samples]),
        datetime_values=torch.stack([s["datetime_values"] for s in samples]),
        boolean_values=torch.stack([s["boolean_values"] for s in samples]),
        text_values=torch.stack([s["text_values"] for s in samples]),
        column_name_values=torch.stack([s["column_name_values"] for s in samples]),
        semantic_types=torch.stack([s["semantic_types"] for s in samples]),
        masks=torch.stack([s["masks"] for s in samples]),
        is_task_node=torch.stack([s["is_task_nodes"] for s in samples]),
        is_padding=is_padding,
        column_attn_mask=col_mask.contiguous(),
        feature_attn_mask=feat_mask.contiguous(),
        neighbor_attn_mask=neighbor_mask.contiguous(),
    )
