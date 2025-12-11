"""
PyTorch Dataset for the Relational Transformer.

Converts RelBench database + task into training samples by:
1. Sampling context windows around task rows using BFS traversal
2. Normalizing numeric values using column statistics
3. Converting cells to Batch format expected by the model

Reference: https://arxiv.org/abs/2510.06377
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset

from dbtransformer.embedder import FrozenEmbedder
from dbtransformer.graph_builder import RelBenchGraphBuilder, Split
from dbtransformer.model import Batch
from dbtransformer.sampler import RelationalSampler
from dbtransformer.sampler_types import Cell, Row, SemanticType


@dataclass
class ColumnStats:
    """Statistics for z-score normalization of a column."""
    mean: float
    std: float
    
    def normalize(self, value: float) -> float:
        """Apply z-score normalization."""
        if self.std == 0 or np.isnan(self.std):
            return 0.0
        return (value - self.mean) / self.std


@dataclass
class DatasetConfig:
    """Configuration for RelationalDataset."""
    seq_len: int = 1024
    width_bound: int = 128  # Max children to sample per row


class RelationalDataset(Dataset):
    """
    PyTorch Dataset for relational transformer.
    
    Each sample is a context window sampled around a task row (seed),
    converted to the Batch format expected by the model.
    """
    
    def __init__(
        self,
        graph_builder: RelBenchGraphBuilder,
        split: Split,
        embedder: FrozenEmbedder,
        config: DatasetConfig | None = None,
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            graph_builder: RelBenchGraphBuilder with the database graph
            split: Which split to use ("train", "val", "test")
            embedder: FrozenEmbedder for text embeddings (optional, will create if needed)
            config: Dataset configuration (optional, will use defaults if not provided)
        """
        self.graph_builder = graph_builder
        self.split = split
        self.config = config or DatasetConfig()
        
        # Initialize sampler
        self.sampler = RelationalSampler(
            context_length=self.config.seq_len,
            width_bound=self.config.width_bound,
        )
        
        # Get task rows for this split (these are our samples)
        self.task_rows = graph_builder.get_task_rows(split)
        logger.info(f"Dataset [{split}]: {len(self.task_rows)} samples")
        
        # Compute column statistics for normalization (use training data only)
        self._column_stats: dict[tuple[str, str], ColumnStats] = {}
        self._global_datetime_stats: ColumnStats | None = None
        self._compute_column_statistics()
        
        # Initialize embedder and pre-compute embeddings
        self.embedder = embedder
        self._column_name_embeddings: dict[tuple[str, str], Float[Tensor, "d"]] = {}
        self._text_value_embeddings: dict[str, Float[Tensor, "d"]] = {}        
        self._precompute_embeddings()
    
    def _compute_column_statistics(self) -> None:
        """
        Compute mean/std for numeric columns from the database.
        
        For datetime columns, we compute GLOBAL statistics across all
        datetime columns to enable cross-table temporal reasoning.
        """
        logger.info("Computing column statistics for normalization...")
        
        all_datetime_values: list[float] = []
        
        for (table_name, col_name), metadata in self.graph_builder.column_metadata.items():
            semantic_type = metadata["semantic_type"]
            
            # Get the table's dataframe
            if table_name.startswith("__task_"):
                # Task table - get from task
                split_name = table_name.replace("__task_", "").replace("__", "")
                task_table = self.graph_builder.task.get_table(split_name)
                df = task_table.df
            else:
                # Regular table
                df = self.graph_builder.db.table_dict[table_name].df
            
            if col_name not in df.columns:
                # TODO(sukwanlee): log warning? This should never happen.
                continue
            
            col_data = df[col_name]
            
            if semantic_type == SemanticType.NUMBER:
                # Standard per-column z-score for numeric
                values = pd.to_numeric(col_data, errors="coerce").dropna()
                if len(values) > 0:
                    self._column_stats[(table_name, col_name)] = ColumnStats(
                        mean=float(values.mean()),
                        std=float(values.std()),
                    )
            
            elif semantic_type == SemanticType.DATETIME:
                # Collect datetime values for global normalization
                dt_values = pd.to_datetime(col_data, errors="coerce").dropna()
                # Convert to seconds since epoch
                sec_values = dt_values.astype(np.int64) / 1e9
                all_datetime_values.extend(sec_values.tolist())
            
            elif semantic_type == SemanticType.BOOLEAN:
                # Boolean: convert to 0/1 then z-score
                bool_values = col_data.dropna().astype(float)
                if len(bool_values) > 0:
                    self._column_stats[(table_name, col_name)] = ColumnStats(
                        mean=float(bool_values.mean()),
                        std=float(bool_values.std()),
                    )
        
        # Compute global datetime statistics
        if all_datetime_values:
            arr = np.array(all_datetime_values, dtype=np.float64)
            self._global_datetime_stats = ColumnStats(
                mean=float(arr.mean()),
                std=float(arr.std()),
            )
        
        logger.info(f"Computed statistics for {len(self._column_stats)} columns")
    
    def _precompute_embeddings(self) -> None:
        """Pre-compute text embeddings for column names and unique text values."""
        logger.info("Pre-computing column name embeddings...")
        
        # Collect all column names: "<column_name> of <table_name>" is the format for schema column
        # embedding in the paper.
        column_names = []
        column_keys = []
        for (table_name, col_name) in self.graph_builder.column_metadata.keys():
            formatted_name = f"{col_name} of {table_name}"
            column_names.append(formatted_name)
            column_keys.append((table_name, col_name))
        
        # Embed column names in batches
        if column_names:
            batch_size = 256
            for i in range(0, len(column_names), batch_size):
                batch_names = column_names[i:i + batch_size]
                batch_keys = column_keys[i:i + batch_size]
                embeddings = self.embedder.embed(batch_names)
                for key, emb in zip(batch_keys, embeddings):
                    self._column_name_embeddings[key] = emb.cpu()
        
        logger.info(f"Embedded {len(self._column_name_embeddings)} column names")
        
        # TODO(sukwanlee): Pre-compute text value embeddings (expensive for large datasets)
    
    def _get_column_name_embedding(self, table_name: str, col_name: str) -> Float[Tensor, "d"]:
        """Get column name embedding."""
        key = (table_name, col_name)
        if key in self._column_name_embeddings:
            return self._column_name_embeddings[key]
        
        formatted_name = f"{col_name} of {table_name}"
        embedding = self.embedder.embed([formatted_name]).cpu()
        self._column_name_embeddings[key] = embedding
        return embedding
    
    def _normalize_value(self, cell: Cell) -> tuple[float, float, float]:
        """
        Normalize a cell's value based on its semantic type.
        
        Returns:
            (number_val, datetime_val, boolean_val) - only one is valid based on type
        """
        value = cell.value
        table = cell.table
        column = cell.column
        semantic_type = cell.semantic_type
        
        # Handle None/NaN values
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0.0, 0.0, 0.0
        
        if semantic_type == SemanticType.NUMBER:
            try:
                num_val = float(value)
                stats = self._column_stats.get((table, column))
                if stats:
                    num_val = stats.normalize(num_val)
                return num_val, 0.0, 0.0
            except (ValueError, TypeError):
                return 0.0, 0.0, 0.0
        
        elif semantic_type == SemanticType.DATETIME:
            try:
                dt = pd.to_datetime(value)
                sec_val = float(dt.value) / 1e9  # Seconds since epoch
                if self._global_datetime_stats:
                    sec_val = self._global_datetime_stats.normalize(sec_val)
                return 0.0, sec_val, 0.0
            except (ValueError, TypeError):
                return 0.0, 0.0, 0.0
        
        elif semantic_type == SemanticType.BOOLEAN:
            try:
                bool_val = float(bool(value))
                stats = self._column_stats.get((table, column))
                if stats:
                    bool_val = stats.normalize(bool_val)
                return 0.0, 0.0, bool_val
            except (ValueError, TypeError):
                return 0.0, 0.0, 0.0
        
        # TEXT type - values go through text_values, not numeric normalization
        return 0.0, 0.0, 0.0
    
    def _get_text_embedding(self, cell: Cell) -> Float[Tensor, "d"]:
        """Get text embedding for a cell's value."""
        if cell.semantic_type != SemanticType.TEXT:
            return torch.zeros(self.embedder.mrl_dimension)
        
        value = cell.value
        if value is None:
            return torch.zeros(self.embedder.mrl_dimension)
        
        str_value = str(value)
        
        # Check cache
        if str_value in self._text_value_embeddings:
            return self._text_value_embeddings[str_value]
        
        emb = self.embedder.embed([str_value])[0].cpu()
        self._text_value_embeddings[str_value] = emb
        return emb
    
    def __len__(self) -> int:
        return len(self.task_rows)
    
    def __getitem__(self, idx: int) -> Batch:
        """
        Sample context around task row and convert to Batch format.
        """
        task_row = self.task_rows[idx]        
        sampled_cells = self.sampler.sample_context(task_row)
        
        # Include the task row's cells (the masked target)
        all_cells: list[Cell] = list(task_row.cells) + sampled_cells
        
        # Build a mapping from row_id to Row object for f2p lookups
        row_by_id: dict[str, Row] = {task_row.row_id: task_row}
        for cell in sampled_cells:
            if cell.row_id not in row_by_id:
                row_obj = self.graph_builder.rows.get(cell.row_id)
                if row_obj:
                    row_by_id[cell.row_id] = row_obj
        # Truncate to config seq_len if needed
        seq_len = self.config.seq_len
        if len(all_cells) > seq_len:
            all_cells = all_cells[:seq_len]
        actual_len = len(all_cells)

        # Build local node index mapping for this sequence
        # Maps global row_id -> local sequence node index
        local_node_mapping: dict[str, int] = {}
        next_local_idx = 0
        
        for cell in all_cells:
            if cell.row_id not in local_node_mapping:
                local_node_mapping[cell.row_id] = next_local_idx
                next_local_idx += 1
        
        # Compute max f2p neighbors for this batch (only count neighbors present in sequence)
        max_f2p_neighbors = 0
        for cell in all_cells:
            row_obj = row_by_id.get(cell.row_id)
            if row_obj:
                count = sum(1 for p in row_obj.f2p_neighbors if p.row_id in local_node_mapping)
                max_f2p_neighbors = max(max_f2p_neighbors, count)
        # Ensure at least 1 to avoid zero-sized dimension
        max_f2p_neighbors = max(max_f2p_neighbors, 1)
        
        # Initialize tensors
        node_indices = torch.zeros(seq_len, dtype=torch.int32)
        table_name_indices = torch.zeros(seq_len, dtype=torch.int32)
        column_name_indices = torch.zeros(seq_len, dtype=torch.int32)
        f2p_neighbor_indices = torch.full((seq_len, max_f2p_neighbors), -1, dtype=torch.int32)
        
        number_values = torch.zeros(seq_len, 1, dtype=torch.float32)
        datetime_values = torch.zeros(seq_len, 1, dtype=torch.float32)
        boolean_values = torch.zeros(seq_len, 1, dtype=torch.float32)
        text_values = torch.zeros(seq_len, self.embedder.mrl_dimension, dtype=torch.float32)
        column_name_values = torch.zeros(seq_len, self.embedder.mrl_dimension, dtype=torch.float32)
        
        semantic_types = torch.zeros(seq_len, dtype=torch.long)
        masks = torch.zeros(seq_len, dtype=torch.bool)
        is_targets = torch.zeros(seq_len, dtype=torch.bool)
        is_task_nodes = torch.zeros(seq_len, dtype=torch.bool)
        is_padding = torch.ones(seq_len, dtype=torch.bool)  # Start as all padding
        class_value_indices = torch.full((seq_len,), -1, dtype=torch.int32)
        
        for pos, cell in enumerate(all_cells):
            is_padding[pos] = False
            
            # Node index (local to this sequence)
            node_indices[pos] = local_node_mapping[cell.row_id]
            
            # Table and column indices (global)
            table_name_indices[pos] = self.graph_builder.get_table_index(cell.table)
            column_name_indices[pos] = self.graph_builder.get_column_index(cell.table, cell.column)
            
            # F2P neighbor indices (local to this sequence)
            row_obj = row_by_id.get(cell.row_id)
            if row_obj:
                neighbor_idx = 0
                for parent in row_obj.f2p_neighbors:
                    if parent.row_id in local_node_mapping:
                        f2p_neighbor_indices[pos, neighbor_idx] = local_node_mapping[parent.row_id]
                        neighbor_idx += 1
            
            # Normalized values
            num_val, dt_val, bool_val = self._normalize_value(cell)
            number_values[pos, 0] = num_val
            datetime_values[pos, 0] = dt_val
            boolean_values[pos, 0] = bool_val
            
            # Text embedding
            text_values[pos] = self._get_text_embedding(cell)
            
            # Column name embedding
            column_name_values[pos] = self._get_column_name_embedding(cell.table, cell.column)
            
            # Semantic type
            semantic_types[pos] = cell.semantic_type.value
            
            # Masking
            masks[pos] = cell.is_masked
            
            # Is this the prediction target? (masked cells in task rows)
            is_targets[pos] = cell.is_masked and cell.row_id == task_row.row_id
            
            # Is this a task node?
            is_task_nodes[pos] = cell.table.startswith("__task_")
        
        # Add batch dimension (1, seq_len, ...)
        return Batch(
            node_indices=node_indices.unsqueeze(0),
            table_name_indices=table_name_indices.unsqueeze(0),
            column_name_indices=column_name_indices.unsqueeze(0),
            f2p_neighbor_indices=f2p_neighbor_indices.unsqueeze(0),
            number_values=number_values.unsqueeze(0),
            datetime_values=datetime_values.unsqueeze(0),
            boolean_values=boolean_values.unsqueeze(0),
            text_values=text_values.unsqueeze(0),
            column_name_values=column_name_values.unsqueeze(0),
            semantic_types=semantic_types.unsqueeze(0),
            masks=masks.unsqueeze(0),
            is_targets=is_targets.unsqueeze(0),
            is_task_nodes=is_task_nodes.unsqueeze(0),
            is_padding=is_padding.unsqueeze(0),
            class_value_indices=class_value_indices.unsqueeze(0),
            true_batch_size=1,
        )
    
    def set_embedder(self, embedder: FrozenEmbedder) -> None:
        """
        Set or update the embedder and recompute embeddings.
        
        Useful for lazy initialization when embedder is expensive to create.
        """
        self.embedder = embedder
        self._precompute_embeddings()
