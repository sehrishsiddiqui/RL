"""
greedy_baseline.py
BFS-based greedy baseline for the ForestFire Helicopter environment.

Strategy:
  1. Find the nearest fire cell (value=2) on the grid via BFS
  2. Navigate toward it using the first step of the shortest path
  3. When already on a fire cell, stay — helicopter extinguishes automatically
  4. If no fire remains, stay (episode effectively won)

Action mapping (matches ForestFireHelicopterEnv):
  0=up-left  1=up  2=up-right
  3=left     4=stay  5=right
  6=down-left  7=down  8=down-right
"""

from collections import deque
import numpy as np


FIRE = 2
STAY = 4

# action -> (delta_row, delta_col)
ACTIONS = {
    0: (-1, -1),
    1: (-1,  0),
    2: (-1,  1),
    3: ( 0, -1),
    4: ( 0,  0),
    5: ( 0,  1),
    6: ( 1, -1),
    7: ( 1,  0),
    8: ( 1,  1),
}

# Movement-only actions (exclude stay)
MOVE_ACTIONS = {k: v for k, v in ACTIONS.items() if k != STAY}


class GreedyBaselineAgent:
    """
    Non-learning baseline agent.
    Accesses the environment grid and agent position directly.
    """

    def act(self, env) -> int:
        """
        Select an action given the current environment state.

        Args:
            env: ForestFireWrapper instance

        Returns:
            action (int in [0, 8])
        """
        grid = env.grid
        pos  = env.agent_pos
        nrows, ncols = grid.shape

        # Already on a fire cell — stay to extinguish
        r, c = pos
        if grid[r, c] == FIRE:
            return STAY

        # Find nearest fire cell and navigate toward it
        target = self._nearest_fire(pos, grid, nrows, ncols)
        if target is None:
            return STAY  # no fire left

        action = self._bfs_first_step(pos, target, nrows, ncols)
        return action if action is not None else STAY

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nearest_fire(self, start, grid, nrows, ncols):
        """BFS from agent position — return coordinates of nearest fire cell."""
        visited = {start}
        queue   = deque([start])

        while queue:
            r, c = queue.popleft()
            if grid[r, c] == FIRE:
                return (r, c)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return None

    def _bfs_first_step(self, start, goal, nrows, ncols):
        """
        BFS from start to goal.
        Returns the action corresponding to the first move along the shortest path.
        """
        if start == goal:
            return STAY

        # visited[(row, col)] = (parent_pos, action_taken_to_get_here)
        visited = {start: None}
        queue   = deque([start])

        while queue:
            r, c = queue.popleft()
            for action, (dr, dc) in MOVE_ACTIONS.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < nrows and 0 <= nc < ncols):
                    continue
                if (nr, nc) in visited:
                    continue
                visited[(nr, nc)] = ((r, c), action)
                if (nr, nc) == goal:
                    return self._trace_first_action(visited, start, goal)
                queue.append((nr, nc))

        return None

    def _trace_first_action(self, visited, start, goal) -> int:
        """Walk parent pointers from goal back to start; return first action."""
        current = goal
        while True:
            parent, action = visited[current]
            if parent == start:
                return action
            current = parent
