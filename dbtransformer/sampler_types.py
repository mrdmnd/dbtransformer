from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class Cell:
    """Represents a single database cell"""

    value: Any
    column: str
    table: str
    row_id: int
    is_masked: bool = False


@dataclass
class Row:
    """Represents a database row with relationships"""

    row_id: str
    table: str
    cells: list[Cell]
    timestamp: datetime | None

    # Foreign key relationships
    f2p_neighbors: list[Row]  # Foreign → Primary (parents; outbound)
    p2f_neighbors: list[Row]  # Primary → Foreign (children; inbound)
