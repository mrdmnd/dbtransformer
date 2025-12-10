import random

from dbtransformer.sampler_types import Cell, Row


class RelationalSampler:
    def __init__(
        self,
        context_length: int = 1024,  # from paper
        width_bound: int = 128,  # tbd
    ) -> None:
        self.context_length = context_length
        self.width_bound = width_bound

    def sample_context(
        self,
        seed_row: Row,
    ) -> list[Cell]:
        """
        Sample context window around seed row.

        Returns list of cells (tokens) for the transformer.
        """
        cells: list[Cell] = []
        frontier: list[Row] = [seed_row]
        to_visit: set[str] = {seed_row.row_id}
        visited: set[str] = set()

        # Track hop distance from seed for prioritization
        hop_distance: dict[str, int] = {seed_row.row_id: 0}
        # Track how each row was reached: "f2p", or "p2f"
        link_type: dict[str, str] = {}

        def add_to_frontier(new_row: Row, from_row: Row, lt: str) -> None:
            frontier.append(new_row)
            to_visit.add(new_row.row_id)
            hop_distance[new_row.row_id] = hop_distance[from_row.row_id] + 1
            link_type[new_row.row_id] = lt

        while len(cells) < self.context_length and frontier:
            # Select next row to explore
            row = self._select_row(frontier, hop_distance, link_type, to_visit)

            # Skip if already visited
            if row.row_id in visited:
                continue
            visited.add(row.row_id)

            # Add all non-missing feature cells from this row
            for cell in row.cells:
                if cell.value is not None and not cell.is_masked:
                    cells.append(cell)
                    if len(cells) >= self.context_length:
                        break

            if len(cells) >= self.context_length:
                break

            # Add F→P neighbors to frontier (always include parents, with temporal filtering)
            for parent in row.f2p_neighbors:
                if (
                    parent.row_id not in visited
                    and parent.row_id not in to_visit
                    and (parent.timestamp is None or seed_row.timestamp is None or parent.timestamp <= seed_row.timestamp)
                ):
                    add_to_frontier(parent, row, "f2p")

            # Add P→F neighbors with temporal filtering and width bound
            valid_children = [
                child
                for child in row.p2f_neighbors
                if (
                    child.row_id not in visited
                    and child.row_id not in to_visit
                    and (child.timestamp is None or seed_row.timestamp is None or child.timestamp <= seed_row.timestamp)
                )
            ]

            # Subsample children (width bound)
            sampled_children = random.sample(valid_children, min(self.width_bound, len(valid_children)))

            for child in sampled_children:
                add_to_frontier(child, row, "p2f")

        return cells

    # TODO(mrdmnd): pull out of class scope, make free fn
    def _select_row(  # noqa: PLR6301
        self,
        frontier: list[Row],
        hop_distance: dict[str, int],
        link_type: dict[str, str],
        to_visit: set[str],
    ) -> Row:
        """
        Select next row to explore.
        Priority: F→P links first, then closest to seed.
        """

        def return_row(row: Row) -> Row:
            frontier.remove(row)
            hop_distance.pop(row.row_id, None)
            link_type.pop(row.row_id, None)
            to_visit.discard(row.row_id)
            return row

        # Prioritize rows added via F→P links over P→F links
        f2p_rows = [row for row in frontier if link_type.get(row.row_id) == "f2p"]

        if f2p_rows:
            return return_row(random.choice(f2p_rows))

        # Otherwise, pick random row closest to seed (single pass)
        closest_rows = []
        min_distance = float("inf")
        for row in frontier:
            dist = hop_distance.get(row.row_id, float("inf"))
            if dist < min_distance:
                min_distance = dist
                closest_rows = [row]
            elif dist == min_distance:
                closest_rows.append(row)

        return return_row(random.choice(closest_rows))
