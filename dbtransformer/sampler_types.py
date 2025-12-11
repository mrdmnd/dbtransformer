from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any


class SemanticType(IntEnum):
    """
    Semantic types for cells - determines which encoder/decoder head to use.
    """

    NUMBER = 0
    TEXT = 1
    DATETIME = 2
    BOOLEAN = 3
    # TODO(mrdmnd): categorical / enum? If you know it has to be one of a few things?
    # TODO(mrdmnd): learn this heuristically?


@dataclass
class Cell:
    """Represents a single database cell"""

    value: Any
    column: str
    table: str
    row_id: str
    semantic_type: SemanticType
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
    
    # Globally unique index for this row.
    node_index: int
