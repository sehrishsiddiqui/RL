"""
train_ppo.py
Train a PPO agent on the ForestFire Helicopter environment using Stable-Baselines3.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.forest_fire_wrapper import ForestFireWrapper


SAVE_DIR = Path(__file__).parent.parent / "results"
SAVE_DIR.mkdir(exist_ok=True)


class RewardLoggerCallback(BaseCallback):
    """Log episode rewards for plotting."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        return True


def train_ppo(total_timesteps: int = 200_000, seed: int = 42):
    print(f"\n=== Training PPO | ForestFireHelicopter5x5-v1 ===")

    env = Monitor(ForestFireWrapper(max_steps=200, seed=seed))
    check_env(env, warn=True)

    eval_env = Monitor(ForestFireWrapper(max_steps=200, seed=seed + 1))

    reward_logger = RewardLoggerCallback()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(SAVE_DIR / "ppo_best"),
        log_path=str(SAVE_DIR / "ppo_eval_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=dict(net_arch=[128, 128]),
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=15,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        verbose=1,
        seed=seed,
        tensorboard_log=str(SAVE_DIR / "tensorboard"),
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger, eval_callback],
        tb_log_name="PPO",
    )

    model.save(str(SAVE_DIR / "ppo_final"))
    np.save(str(SAVE_DIR / "ppo_episode_rewards.npy"), np.array(reward_logger.episode_rewards))
    print(f"PPO model saved to {SAVE_DIR / 'ppo_final'}")

    return model, reward_logger.episode_rewards


if __name__ == "__main__":
    train_ppo()
