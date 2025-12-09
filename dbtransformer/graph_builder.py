import pandas as pd
from typing import Dict, List, Literal, Optional
from loguru import logger
from relbench.base import Dataset, Database
from relbench.tasks import BaseTask
from dbtransformer.types import Cell, Row

Split = Literal["train", "val", "test"]


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
        
        # Task rows keyed by split -> list of Row
        self._task_rows: Dict[Split, List[Row]] = {}
        
        self._build_rows()
        self._build_relationships()
    
    def _build_rows(self) -> None:
        """Create Row objects for each database row."""
        for table_name, table in self.db.table_dict.items():
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
            
            for idx, row in table.df.iterrows():
                # Generate unique global row ID
                global_row_id = self._make_row_id(table_name, pkey_col, row, idx)
                
                # Build Cell objects for feature columns
                cells = [
                    Cell(
                        value=row[col],
                        column=col,
                        table=table_name,
                        row_id=global_row_id,
                    )
                    for col in feature_cols
                ]
                
                # Extract timestamp from time_col
                timestamp = row[time_col] if time_col is not None else None
                
                row_obj = Row(
                    row_id=global_row_id,
                    table=table_name,
                    cells=cells,
                    timestamp=timestamp,
                    f2p_neighbors=[],
                    p2f_neighbors=[],
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
        
        for idx, row in task_df.iterrows():
            global_row_id = f"{task_table_name}:{idx}"
            
            timestamp = row[time_col]
            # Create masked label cell
            label_cell = Cell(
                value=row[target_col],
                column=target_col,
                table=task_table_name,
                row_id=global_row_id,
                is_masked=True,  # Label is masked during inference
            )
            
            # Create task row
            task_row = Row(
                row_id=global_row_id,
                table=task_table_name,
                cells=[label_cell],
                timestamp=timestamp,
                f2p_neighbors=[],
                p2f_neighbors=[],
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