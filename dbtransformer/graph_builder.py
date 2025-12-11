import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Optional
from loguru import logger
from relbench.base import Dataset, Database
from relbench.tasks import BaseTask
from dbtransformer.sampler_types import Cell, Row, SemanticType

Split = Literal["train", "val", "test"]


def infer_semantic_type(dtype: np.dtype) -> SemanticType:
    """Infer semantic type from pandas dtype and value."""
    dtype_str = str(dtype)
    
    # Boolean types
    if dtype_str == "bool" or dtype_str == "boolean":
        return SemanticType.BOOLEAN
    
    # Datetime types
    if "datetime" in dtype_str or "timestamp" in dtype_str:
        return SemanticType.DATETIME
    
    # Numeric types
    if np.issubdtype(dtype, np.number):
        return SemanticType.NUMBER
    
    # Object/string types - treat as text
    if dtype_str == "object" or dtype_str == "string" or "str" in dtype_str:
        return SemanticType.TEXT
    
    # Category - treat as text (will use class_value_indices for prediction)
    if dtype_str == "category":
        return SemanticType.TEXT
    
    # Default to TEXT for unknown types
    return SemanticType.TEXT


class RelBenchGraphBuilder:
    """
    Converts a RelBench database + task into a traversable Row/Cell graph.
    
    The graph includes:
    - All database rows (from db.table_dict)
    - Task rows (from task.get_table) linked to their entity rows
    """
    
    def __init__(self, dataset: Dataset, task: BaseTask):
        self.dataset = dataset
        self.task = task
        self.db: Database = dataset.get_db()
        
        # Index for O(1) parent lookups: (table_name, pkey_value) -> row_id
        self._pkey_index: Dict[tuple, str] = {}
        
        # Rows keyed by global row_id
        self.rows: Dict[str, Row] = {}
        
        # Sequential node index counter
        self._next_node_index: int = 0
        
        # Task rows keyed by split -> list of Row
        self._task_rows: Dict[Split, List[Row]] = {}
        
        # Column metadata: (table, column) -> {"dtype": dtype, "semantic_type": SemanticType}
        self.column_metadata: Dict[tuple[str, str], dict] = {}
        
        # Unique table name -> index mapping (for attention masks)
        self._table_name_to_idx: Dict[str, int] = {}
        
        # Unique (table, column) -> index mapping (for attention masks)
        self._column_to_idx: Dict[tuple[str, str], int] = {}
        
        self._build_rows()
        self._build_relationships()
    
    def _build_rows(self) -> None:
        """Create Row objects for each database row."""
        for table_name, table in self.db.table_dict.items():
            # Assign table index
            if table_name not in self._table_name_to_idx:
                self._table_name_to_idx[table_name] = len(self._table_name_to_idx)
            
            pkey_col = table.pkey_col
            time_col = table.time_col
            
            if time_col is None:
                logger.warning(
                    f"Table '{table_name}' has no time_col. "
                    "Rows will have timestamp=None."
                )
            
            # Identify feature columns (exclude keys and time)
            key_cols = self._get_key_columns(table_name)
            feature_cols = [
                col for col in table.df.columns
                if col not in key_cols and col != time_col
            ]
            
            # Build column metadata for feature columns
            for col in feature_cols:
                col_key = (table_name, col)
                if col_key not in self._column_to_idx:
                    self._column_to_idx[col_key] = len(self._column_to_idx)
                
                dtype = table.df[col].dtype
                semantic_type = infer_semantic_type(dtype)
                self.column_metadata[col_key] = {
                    "dtype": dtype,
                    "semantic_type": semantic_type,
                }
            
            for idx, row in table.df.iterrows():
                # Generate unique global row ID
                global_row_id = self._make_row_id(table_name, pkey_col, row, idx)
                
                # Build Cell objects for feature columns
                cells = []
                for col in feature_cols:
                    col_key = (table_name, col)
                    semantic_type = self.column_metadata[col_key]["semantic_type"]
                    cells.append(
                        Cell(
                            value=row[col],
                            column=col,
                            table=table_name,
                            row_id=global_row_id,
                            semantic_type=semantic_type,
                        )
                    )
                
                # Extract timestamp from time_col
                timestamp = row[time_col] if time_col is not None else None
                
                # Assign sequential node index
                node_index = self._next_node_index
                self._next_node_index += 1
                
                row_obj = Row(
                    row_id=global_row_id,
                    table=table_name,
                    cells=cells,
                    timestamp=timestamp,
                    f2p_neighbors=[],
                    p2f_neighbors=[],
                    node_index=node_index,
                )
                
                self.rows[global_row_id] = row_obj
                
                # Index by primary key for efficient parent lookups
                if pkey_col is not None:
                    pkey_value = row[pkey_col]
                    self._pkey_index[(table_name, pkey_value)] = global_row_id
    
    def _build_relationships(self) -> None:
        """Build F→P and P→F links between rows."""
        for table_name, table in self.db.table_dict.items():
            pkey_col = table.pkey_col
            fkey_mapping = table.fkey_col_to_pkey_table
            
            if not fkey_mapping:
                continue
            
            for idx, row in table.df.iterrows():
                child_row_id = self._make_row_id(table_name, pkey_col, row, idx)
                child = self.rows.get(child_row_id)
                if child is None:
                    continue
                
                # Process each foreign key column
                for fk_col, parent_table in fkey_mapping.items():
                    fk_value = row[fk_col]
                    
                    # Skip null foreign keys
                    if pd.isna(fk_value):
                        continue
                    
                    # Look up parent via index
                    parent_row_id = self._pkey_index.get((parent_table, fk_value))
                    if parent_row_id is None:
                        continue
                    
                    parent = self.rows.get(parent_row_id)
                    if parent is None:
                        continue
                    
                    # F→P: child references parent
                    child.f2p_neighbors.append(parent)
                    
                    # P→F: parent is referenced by child
                    parent.p2f_neighbors.append(child)
    
    def _get_key_columns(self, table_name: str) -> List[str]:
        """Get primary and foreign key column names to exclude from features."""
        table = self.db.table_dict[table_name]
        cols = []
        if table.pkey_col is not None:
            cols.append(table.pkey_col)
        cols.extend(table.fkey_col_to_pkey_table.keys())
        return cols
    
    def _make_row_id(
        self,
        table_name: str,
        pkey_col: Optional[str],
        row: pd.Series,
        idx: int,
    ) -> str:
        """Generate a globally unique row ID."""
        if pkey_col is not None:
            return f"{table_name}:{row[pkey_col]}"
        # Fallback to index for tables without primary key
        return f"{table_name}:idx:{idx}"
    
    def get_row(self, table_name: str, pkey_value) -> Optional[Row]:
        """Retrieve a Row by table name and primary key value."""
        row_id = self._pkey_index.get((table_name, pkey_value))
        return self.rows.get(row_id) if row_id else None
    
    def get_task_rows(self, split: Split) -> List[Row]:
        """
        Get task rows for a split (train/val/test).
        
        Task rows are the seed rows for sampling. Each task row:
        - Has a timestamp (prediction time)
        - Links to an entity row via foreign key
        - Contains a masked label cell (the prediction target)
        
        Returns:
            List of task Row objects for the specified split.
        """
        if split not in self._task_rows:
            self._build_task_rows(split)
        return self._task_rows[split]
    
    def _build_task_rows(self, split: Split) -> None:
        """Build task rows for a specific split and link to entity rows."""
        task_table = self.task.get_table(split)
        task_df = task_table.df
        
        entity_table = self.task.entity_table
        entity_col = self.task.entity_col
        time_col = self.task.time_col
        target_col = self.task.target_col
        
        task_table_name = f"__task_{split}__"
        task_rows = []
        
        # Register task table in indices
        if task_table_name not in self._table_name_to_idx:
            self._table_name_to_idx[task_table_name] = len(self._table_name_to_idx)
        
        # Infer semantic type for target column
        target_dtype = task_df[target_col].dtype
        target_semantic_type = infer_semantic_type(target_dtype)
        
        # Register column metadata for task target
        col_key = (task_table_name, target_col)
        if col_key not in self._column_to_idx:
            self._column_to_idx[col_key] = len(self._column_to_idx)
        self.column_metadata[col_key] = {
            "dtype": target_dtype,
            "semantic_type": target_semantic_type,
        }
        
        for idx, row in task_df.iterrows():
            global_row_id = f"{task_table_name}:{idx}"
            
            timestamp = row[time_col]
            
            # Create masked label cell with semantic type
            label_cell = Cell(
                value=row[target_col],
                column=target_col,
                table=task_table_name,
                row_id=global_row_id,
                semantic_type=target_semantic_type,
                is_masked=True,  # Label is masked during inference
            )
            
            # Assign sequential node index for task row
            node_index = self._next_node_index
            self._next_node_index += 1
            # Create task row
            task_row = Row(
                row_id=global_row_id,
                table=task_table_name,
                cells=[label_cell],
                timestamp=timestamp,
                f2p_neighbors=[],
                p2f_neighbors=[],
                node_index=node_index,
            )
            
            # Link task row to entity row (F→P)
            entity_key = row[entity_col]
            entity_row = self.get_row(entity_table, entity_key)
            
            if entity_row is not None:
                task_row.f2p_neighbors.append(entity_row)
                # Don't add P→F back-link (task rows are entry points, not traversed to)
            else:
                logger.warning(
                    f"Task row {idx} references missing entity "
                    f"{entity_table}:{entity_key}"
                )
            
            task_rows.append(task_row)
        
        self._task_rows[split] = task_rows
        logger.info(f"Built {len(task_rows)} task rows for split '{split}'")
    
    def get_table_index(self, table_name: str) -> int:
        """Get the unique integer index for a table name."""
        return self._table_name_to_idx.get(table_name, -1)
    
    def get_column_index(self, table_name: str, column_name: str) -> int:
        """Get the unique integer index for a (table, column) pair."""
        return self._column_to_idx.get((table_name, column_name), -1)
    
    @property
    def num_tables(self) -> int:
        """Total number of unique tables."""
        return len(self._table_name_to_idx)
    
    @property
    def num_columns(self) -> int:
        """Total number of unique (table, column) pairs."""
        return len(self._column_to_idx)