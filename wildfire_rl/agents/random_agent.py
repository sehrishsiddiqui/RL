"""
random_agent.py
Trivial random baseline — selects a uniformly random action each step.

This agent establishes the absolute floor for performance. Any learning
agent that cannot beat random is not learning anything useful.
"""

import numpy as np


class RandomAgent:
    """Selects a random action from the environment's action space each step."""

    def __init__(self, n_actions: int = 9, seed: int = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, env) -> int:
        return self.rng.integers(0, self.n_actions)
