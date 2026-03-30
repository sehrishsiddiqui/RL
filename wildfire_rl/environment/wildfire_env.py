"""
wildfire_env.py
Custom Gymnasium environment for wildfire containment.

The agent is a fire incident commander routing suppression resources
across a terrain grid to establish containment lines before fire
reaches populated zones.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.fire_spread import FireSpread, WIND_VECTORS


# Action constants
MOVE_N = 0
MOVE_S = 1
MOVE_E = 2
MOVE_W = 3
DEPLOY  = 4
HOLD    = 5

ACTION_DELTAS = {
    MOVE_N: (-1, 0),
    MOVE_S: (1, 0),
    MOVE_E: (0, 1),
    MOVE_W: (0, -1),
}

# Reward constants
REWARD_PER_STEP           = -1
REWARD_CONTAINMENT_PLACED = 50
REWARD_FULL_CONTAINMENT   = 200
REWARD_FIRE_REACHES_ZONE  = -100
REWARD_WASTED_DEPLOY      = -20


class WildfireEnv(gym.Env):
    """
    Wildfire Containment Environment.

    State space (flattened observation vector):
        - Agent position:       2 values  (row, col) normalized to [0,1]
        - Fire grid (flattened): grid_size^2 values, each in {0,1,2,3} normalized
        - Wind direction:        4 values  (one-hot: N, S, E, W)
        - Resources remaining:   1 value   normalized to [0,1]

    Action space: Discrete(6)
        0 = Move North
        1 = Move South
        2 = Move East
        3 = Move West
        4 = Deploy containment resource at current cell
        5 = Hold (do nothing)

    Episode ends when:
        - Fire reaches a populated zone cell (failure)
        - Fire is fully contained (success)
        - Max timesteps reached (timeout)
        - Agent runs out of resources and fire is still active
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = 12,
        wind_direction: str = "S",
        base_spread_prob: float = 0.35,
        wind_boost: float = 0.3,
        max_resources: int = 15,
        max_steps: int = 200,
        num_populated_zones: int = 3,
        render_mode: str = None,
        seed: int = None,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.wind_direction = wind_direction
        self.max_resources = max_resources
        self.max_steps = max_steps
        self.num_populated_zones = num_populated_zones
        self.render_mode = render_mode
        self.seed_val = seed

        # Fire spread engine
        self.fire = FireSpread(
            grid_size=grid_size,
            wind_direction=wind_direction,
            base_spread_prob=base_spread_prob,
            wind_boost=wind_boost,
            seed=seed,
        )

        # Observation space — flattened vector
        obs_size = (
            2                    # agent position (normalized)
            + grid_size * grid_size  # fire grid
            + 4                  # wind one-hot
            + 1                  # resources remaining (normalized)
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Action space
        self.action_space = spaces.Discrete(6)

        # Internal state (set in reset)
        self.agent_pos = None
        self.resources = None
        self.populated_zones = None
        self.current_step = None
        self.rng = np.random.default_rng(seed)

        # Pygame rendering
        self._window = None
        self._clock = None
        self.cell_size = 50

    # ------------------------------------------------------------------
    # Core Gym Methods
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.resources = self.max_resources

        # Place agent in a random corner
        corners = [(0, 0), (0, self.grid_size-1),
                   (self.grid_size-1, 0), (self.grid_size-1, self.grid_size-1)]
        self.agent_pos = list(corners[self.rng.integers(0, 4)])

        # Place fire start at center of grid
        fire_start = (self.grid_size // 2, self.grid_size // 2)
        self.fire.reset(fire_start)

        # Place populated zones — fixed positions along one edge opposite to fire start
        self.populated_zones = self._generate_populated_zones()

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        self.current_step += 1
        reward = REWARD_PER_STEP
        terminated = False
        truncated = False

        # --- Agent action ---
        if action in ACTION_DELTAS:
            self._move_agent(action)
        elif action == DEPLOY:
            reward += self._deploy_resource()
        elif action == HOLD:
            pass  # nothing happens, step penalty already applied

        # --- Fire spreads ---
        self.fire.step()

        # --- Check if fire reached populated zones ---
        for zone in self.populated_zones:
            if self.fire.is_burning(zone[0], zone[1]) or self.fire.fire_grid[zone[0], zone[1]] == 2:
                reward += REWARD_FIRE_REACHES_ZONE
                terminated = True

        # --- Check full containment ---
        if self.fire.is_fully_contained():
            reward += REWARD_FULL_CONTAINMENT
            terminated = True

        # --- Timeout ---
        if self.current_step >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {
            "step": self.current_step,
            "resources": self.resources,
            "active_fire": self.fire.active_fire_count(),
            "burned_cells": self.fire.burned_count(),
            "agent_pos": tuple(self.agent_pos),
        }

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def close(self):
        if self._window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _move_agent(self, action: int):
        dr, dc = ACTION_DELTAS[action]
        nr = self.agent_pos[0] + dr
        nc = self.agent_pos[1] + dc

        # Stay in bounds
        nr = max(0, min(self.grid_size - 1, nr))
        nc = max(0, min(self.grid_size - 1, nc))

        # Can only move to passable cells (not burned)
        if self.fire.is_passable(nr, nc):
            self.agent_pos = [nr, nc]

    def _deploy_resource(self) -> float:
        if self.resources <= 0:
            return 0.0

        r, c = self.agent_pos
        fire_val = self.fire.fire_grid[r, c]

        if fire_val == 2:
            # Deploying on already burned cell — wasted
            self.resources -= 1
            return REWARD_WASTED_DEPLOY

        if fire_val == 0:
            # Valid placement
            self.fire.place_containment(r, c)
            self.resources -= 1
            return REWARD_CONTAINMENT_PLACED

        # On fire or already containment — no effect
        return 0.0

    def _generate_populated_zones(self) -> list:
        """Place populated zones along the bottom edge of the grid."""
        zones = []
        cols = np.linspace(1, self.grid_size - 2, self.num_populated_zones, dtype=int)
        for col in cols:
            zones.append((self.grid_size - 1, int(col)))
        return zones

    def _get_obs(self) -> np.ndarray:
        # Agent position normalized
        agent_norm = np.array([
            self.agent_pos[0] / (self.grid_size - 1),
            self.agent_pos[1] / (self.grid_size - 1),
        ], dtype=np.float32)

        # Fire grid normalized (values 0–3 → 0–1)
        fire_norm = (self.fire.get_fire_map() / 3.0).flatten().astype(np.float32)

        # Wind one-hot
        wind_order = ["N", "S", "E", "W"]
        wind_onehot = np.zeros(4, dtype=np.float32)
        if self.wind_direction in wind_order:
            wind_onehot[wind_order.index(self.wind_direction)] = 1.0

        # Resources normalized
        resources_norm = np.array(
            [self.resources / self.max_resources], dtype=np.float32
        )

        return np.concatenate([agent_norm, fire_norm, wind_onehot, resources_norm])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_frame(self):
        try:
            import pygame
        except ImportError:
            print("pygame not installed — skipping render. Run: pip install pygame")
            return None

        window_size = self.grid_size * self.cell_size

        if self._window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode((window_size, window_size))
            pygame.display.set_caption("Wildfire Containment RL")
            self._clock = pygame.time.Clock()

        canvas = pygame.Surface((window_size, window_size))
        canvas.fill((200, 200, 200))  # background gray

        # Color map
        colors = {
            0: (34, 139, 34),    # unburned — forest green
            1: (255, 69, 0),     # on fire — orange-red
            2: (50, 50, 50),     # burned — dark gray
            3: (139, 90, 43),    # containment line — brown
        }

        fire_grid = self.fire.get_fire_map()

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                color = colors[fire_grid[r, c]]
                rect = pygame.Rect(
                    c * self.cell_size, r * self.cell_size,
                    self.cell_size - 1, self.cell_size - 1
                )
                pygame.draw.rect(canvas, color, rect)

        # Draw populated zones — blue
        for (zr, zc) in self.populated_zones:
            rect = pygame.Rect(
                zc * self.cell_size, zr * self.cell_size,
                self.cell_size - 1, self.cell_size - 1
            )
            pygame.draw.rect(canvas, (0, 100, 255), rect)

        # Draw agent — white circle
        ar, ac = self.agent_pos
        center = (
            ac * self.cell_size + self.cell_size // 2,
            ar * self.cell_size + self.cell_size // 2,
        )
        pygame.draw.circle(canvas, (255, 255, 255), center, self.cell_size // 3)

        if self.render_mode == "human":
            self._window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self._clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
