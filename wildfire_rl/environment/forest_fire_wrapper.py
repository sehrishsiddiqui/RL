"""
forest_fire_wrapper.py
SB3-compatible wrapper around gym-cellular-automata's ForestFireHelicopter5x5-v1.

Key changes vs the original custom WildfireEnv:
  - 5x5 grid (was 12x12) — simpler, faster to learn
  - Helicopter extinguishes fire by flying over it (no containment line mechanic)
  - Unlimited extinguishing capacity (no resource limit)
  - No burned/impassable cells — grid values are 0=empty, 1=tree, 2=fire
  - Episodes truncate at max_steps (env never terminates naturally)
  - Success = no fire cells remaining

Observation vector (30 values, all normalized to [0, 1]):
  [0:25]  grid flattened, divided by 2.0
  [25:27] ca_params [p_fire, p_tree] — already in [0, 1]
  [27:29] helicopter position [row/nrows, col/ncols]
  [29]    freeze countdown / max_freeze
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import gym_cellular_automata  # noqa: F401 — registers ForestFireHelicopter5x5-v1


EMPTY = 0
TREE  = 1
FIRE  = 2

ENV_ID = "ForestFireHelicopter5x5-v1"


class ForestFireWrapper(gym.Wrapper):
    """
    Wraps ForestFireHelicopter5x5-v1 for use with Stable-Baselines3:
      - Flattens the Tuple observation space into a 1D float32 Box
      - Adds max-step truncation
      - Exposes .grid and .agent_pos for the greedy baseline
    """

    def __init__(self, max_steps: int = 200, seed: int = None):
        base_env = gym.make(ENV_ID)
        super().__init__(base_env)

        self._inner = base_env.unwrapped  # unwrap all wrappers to reach core env
        self.nrows = self._inner.nrows
        self.ncols = self._inner.ncols
        self._max_steps = max_steps
        self._max_freeze = self._inner._max_freeze
        self._current_step = 0
        self._prev_fire_count = 0
        self._max_manhattan = self.nrows + self.ncols - 2

        # 25 (grid) + 2 (ca_params) + 2 (position) + 1 (freeze)
        #   + 4 (fire awareness: dist, direction_r, direction_c, fire_count) = 34
        obs_size = self.nrows * self.ncols + 2 + 2 + 1 + 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        # Action space inherited: Discrete(9)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._current_step = 0
        self._prev_fire_count = int(np.sum(self._inner.grid == FIRE))
        return self._flatten(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_step += 1

        grid = self._inner.grid
        fire_remaining = int(np.sum(grid == FIRE))

        info["fire_remaining"] = fire_remaining
        info["step"] = self._current_step
        info["success"] = fire_remaining == 0
        info["trees_remaining"] = int(np.sum(grid == TREE))

        # Reward shaping: the base reward (trees - fires)/25 is dominated
        # by stochastic fire spawning. We layer on action-driven signals
        # with fire extinguishing as the dominant incentive.
        shaped_reward = float(reward)

        # Dominant signal: big reward for extinguishing a fire cell
        if info.get("hit", False):
            shaped_reward += 10.0

        # Reward net fire reduction (captures multi-cell extinguishing)
        fire_delta = self._prev_fire_count - fire_remaining
        if fire_delta > 0:
            shaped_reward += fire_delta * 2.0

        # Bonus for achieving zero fire
        if fire_remaining == 0:
            shaped_reward += 5.0

        # Per-step urgency: small penalty proportional to active fires
        shaped_reward -= fire_remaining * 0.1

        self._prev_fire_count = fire_remaining

        if self._current_step >= self._max_steps:
            truncated = True

        return self._flatten(obs), shaped_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flatten(self, obs) -> np.ndarray:
        grid, (ca_params, position, freeze) = obs

        grid_norm   = grid.flatten().astype(np.float32) / 2.0
        ca_norm     = ca_params.astype(np.float32)
        pos_norm    = np.array([
            position[0] / max(self.nrows - 1, 1),
            position[1] / max(self.ncols - 1, 1),
        ], dtype=np.float32)
        freeze_norm = np.array(
            [float(freeze) / max(self._max_freeze, 1)], dtype=np.float32
        )

        # Fire-awareness features: give the agent a "compass" toward fire
        fire_positions = np.argwhere(grid == FIRE)
        if len(fire_positions) > 0:
            pos_arr = np.array([position[0], position[1]])
            dists = np.abs(fire_positions - pos_arr).sum(axis=1)
            nearest = fire_positions[np.argmin(dists)]
            direction = nearest - pos_arr

            fire_features = np.array([
                float(dists.min()) / self._max_manhattan,
                float(direction[0]) / max(self.nrows - 1, 1),
                float(direction[1]) / max(self.ncols - 1, 1),
                float(len(fire_positions)) / (self.nrows * self.ncols),
            ], dtype=np.float32)
        else:
            fire_features = np.zeros(4, dtype=np.float32)

        return np.concatenate([grid_norm, ca_norm, pos_norm, freeze_norm, fire_features])

    # ------------------------------------------------------------------
    # Properties for greedy baseline direct access
    # ------------------------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        """Current 5x5 fire grid (0=empty, 1=tree, 2=fire)."""
        return self._inner.grid

    @property
    def agent_pos(self) -> tuple:
        """Helicopter position as (row, col)."""
        _, (_, position, _) = self._inner.state
        return (int(position[0]), int(position[1]))
