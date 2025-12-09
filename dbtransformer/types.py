from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class Cell:
    """Represents a single database cell"""
    value: any
    column: str
    table: str
    row_id: int
    is_masked: bool = False

@dataclass
class Row:
    """Represents a database row with relationships"""
    row_id: str
    table: str
    cells: List[Cell]
    timestamp: Optional[datetime]
    
    # Foreign key relationships
    f2p_neighbors: List[Row]  # Foreign → Primary (parents; outbound)
    p2f_neighbors: List[Row]  # Primary → Foreign (children; inbound)