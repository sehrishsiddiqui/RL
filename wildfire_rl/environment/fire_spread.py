"""
fire_spread.py
Cellular automaton module for wildfire spread simulation.

Each timestep, fire propagates to adjacent unburned cells based on
wind direction and a base spread probability. Burned cells are permanent.
"""

import numpy as np


# Wind direction vectors: (row_delta, col_delta)
WIND_VECTORS = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
    "none": (0, 0),
}


class FireSpread:
    """
    Manages fire state on a 2D grid.

    Grid cell values:
        0 = unburned (passable)
        1 = on fire (spreading this timestep)
        2 = burned out (permanently impassable)
        3 = containment line (blocks spread, placed by agent)
    """

    def __init__(
        self,
        grid_size: int,
        wind_direction: str = "N",
        base_spread_prob: float = 0.4,
        wind_boost: float = 0.3,
        seed: int = None,
    ):
        """
        Args:
            grid_size:        Size of the square grid (grid_size x grid_size)
            wind_direction:   Cardinal direction wind is blowing TOWARD ("N","S","E","W","none")
            base_spread_prob: Probability fire spreads to any adjacent cell
            wind_boost:       Additional spread probability in wind direction
            seed:             Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.wind_direction = wind_direction
        self.wind_vector = WIND_VECTORS[wind_direction]
        self.base_spread_prob = base_spread_prob
        self.wind_boost = wind_boost
        self.rng = np.random.default_rng(seed)

        # Fire grid — initialized in reset()
        self.fire_grid = np.zeros((grid_size, grid_size), dtype=np.int32)

    def reset(self, fire_start: tuple):
        """
        Reset fire grid. Place initial fire at fire_start cell.

        Args:
            fire_start: (row, col) of initial fire cell
        """
        self.fire_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.fire_grid[fire_start[0], fire_start[1]] = 1

    def step(self):
        """
        Advance fire spread by one timestep.
        - Cells on fire (1) spread to adjacent unburned (0) cells
        - Cells on fire (1) become burned out (2) after spreading
        - Containment lines (3) block spread

        Returns:
            new_fire_cells: list of (row, col) that newly caught fire
        """
        new_fire_grid = self.fire_grid.copy()
        new_fire_cells = []

        # Find all currently burning cells
        burning_cells = list(zip(*np.where(self.fire_grid == 1)))

        for (r, c) in burning_cells:
            # Spread to 4 adjacent neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc

                # Out of bounds check
                if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                    continue

                # Only spread to unburned cells (not containment lines)
                if self.fire_grid[nr, nc] != 0:
                    continue

                # Compute spread probability — boosted in wind direction
                spread_prob = self.base_spread_prob
                if (dr, dc) == self.wind_vector:
                    spread_prob = min(1.0, spread_prob + self.wind_boost)

                if self.rng.random() < spread_prob:
                    new_fire_grid[nr, nc] = 1
                    new_fire_cells.append((nr, nc))

            # Current burning cell becomes burned out
            new_fire_grid[r, c] = 2

        self.fire_grid = new_fire_grid
        return new_fire_cells

    def place_containment(self, row: int, col: int) -> bool:
        """
        Place a containment line at (row, col).
        Only valid on unburned cells (value 0).

        Returns:
            True if successfully placed, False if cell is already burned/on fire
        """
        if self.fire_grid[row, col] == 0:
            self.fire_grid[row, col] = 3
            return True
        return False

    def is_burning(self, row: int, col: int) -> bool:
        return self.fire_grid[row, col] == 1

    def is_passable(self, row: int, col: int) -> bool:
        """Agent can only move through unburned (0) or containment (3) cells."""
        return self.fire_grid[row, col] in (0, 3)

    def is_fully_contained(self) -> bool:
        """Fire is contained when no cells are currently burning."""
        return not np.any(self.fire_grid == 1)

    def get_fire_map(self) -> np.ndarray:
        """Return a copy of the current fire grid."""
        return self.fire_grid.copy()

    def active_fire_count(self) -> int:
        return int(np.sum(self.fire_grid == 1))

    def burned_count(self) -> int:
        return int(np.sum(self.fire_grid == 2))
