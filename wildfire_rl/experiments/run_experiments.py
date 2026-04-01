"""
run_experiments.py
Complete experiment pipeline: train, evaluate, hyperparameter sweep.

Usage:
    python experiments/run_experiments.py                  # full pipeline
    python experiments/run_experiments.py --eval-only      # skip training, just evaluate
    python experiments/run_experiments.py --sweep-only     # only run hyperparam sweep
"""

import sys
import time
import pickle
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.utils import get_linear_fn

from environment.forest_fire_wrapper import ForestFireWrapper
from agents.random_agent import RandomAgent
from agents.greedy_baseline import GreedyBaselineAgent


RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
N_EVAL_EPISODES = 100
MAX_STEPS = 200
TOTAL_TIMESTEPS = 1_000_000


# =====================================================================
# Callbacks
# =====================================================================

class RewardLoggerCallback(BaseCallback):
    """Track episode rewards during training."""

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


# =====================================================================
# Training
# =====================================================================

def train_agent(algo_name: str, seed: int = SEED, total_timesteps: int = TOTAL_TIMESTEPS,
                **override_params):
    """
    Train a single agent. Returns (model, episode_rewards, wall_time_seconds).
    """
    env = Monitor(ForestFireWrapper(max_steps=MAX_STEPS, seed=seed))
    eval_env = Monitor(ForestFireWrapper(max_steps=MAX_STEPS, seed=seed + 1))

    reward_logger = RewardLoggerCallback()

    save_prefix = algo_name.lower()
    if override_params:
        # Tag with overridden params for sweep runs
        tag = "_".join(f"{k}{v}" for k, v in override_params.items())
        save_prefix = f"{algo_name.lower()}_{tag}"

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / f"{save_prefix}_best"),
        log_path=str(RESULTS_DIR / f"{save_prefix}_eval_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )

    if algo_name == "DQN":
        params = dict(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=5000,
            batch_size=256,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_final_eps=0.02,
            target_update_interval=1000,
            train_freq=4,
            verbose=0,
            seed=seed,
            tensorboard_log=str(RESULTS_DIR / "tensorboard"),
        )
        params.update(override_params)
        model = DQN(**params)

    elif algo_name == "PPO":
        params = dict(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
            learning_rate=get_linear_fn(3e-4, 1e-5, 1.0),
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=seed,
            tensorboard_log=str(RESULTS_DIR / "tensorboard"),
        )
        params.update(override_params)
        model = PPO(**params)
    else:
        raise ValueError(f"Unknown algo: {algo_name}")

    print(f"  Training {save_prefix} for {total_timesteps:,} timesteps...")
    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger, eval_callback],
        tb_log_name=save_prefix,
    )
    wall_time = time.time() - start

    model.save(str(RESULTS_DIR / f"{save_prefix}_final"))
    np.save(str(RESULTS_DIR / f"{save_prefix}_episode_rewards.npy"),
            np.array(reward_logger.episode_rewards))

    print(f"  {save_prefix} done in {wall_time:.1f}s ({len(reward_logger.episode_rewards)} episodes)")

    env.close()
    eval_env.close()

    return model, reward_logger.episode_rewards, wall_time


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_agent(agent_name: str, get_action_fn, n_episodes: int = N_EVAL_EPISODES,
                   seed: int = SEED):
    """
    Evaluate any agent. Returns dict of per-episode metrics.

    Args:
        agent_name:    label for this agent
        get_action_fn: callable(obs, env) -> action
        n_episodes:    number of evaluation episodes
        seed:          random seed
    """
    env = ForestFireWrapper(max_steps=MAX_STEPS, seed=seed)

    results = {
        "rewards": [],
        "successes": [],
        "steps": [],
        "fire_remaining": [],
        "trees_remaining": [],
        "position_counts": np.zeros((env.nrows, env.ncols), dtype=np.int32),
        "fire_over_time": [],   # list of arrays, one per episode
    }

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        info = {}
        fire_trace = []

        while not done:
            action = get_action_fn(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Track position
            r, c = env.agent_pos
            results["position_counts"][r, c] += 1

            # Track fire remaining
            fire_trace.append(info.get("fire_remaining", 0))

        results["rewards"].append(total_reward)
        results["successes"].append(1 if info.get("success", False) else 0)
        results["steps"].append(info.get("step", 0))
        results["fire_remaining"].append(info.get("fire_remaining", -1))
        results["trees_remaining"].append(info.get("trees_remaining", 0))
        results["fire_over_time"].append(fire_trace)

    env.close()

    # Summary stats
    results["mean_reward"] = float(np.mean(results["rewards"]))
    results["std_reward"] = float(np.std(results["rewards"]))
    results["success_rate"] = float(np.mean(results["successes"]) * 100)
    results["mean_steps"] = float(np.mean(results["steps"]))
    results["mean_fire_remaining"] = float(np.mean(results["fire_remaining"]))

    print(f"  {agent_name:>10s} | Reward: {results['mean_reward']:>8.3f} ± {results['std_reward']:.3f} | "
          f"Success: {results['success_rate']:>5.1f}% | Fire left: {results['mean_fire_remaining']:.2f}")

    return results


# =====================================================================
# Hyperparameter Sweep
# =====================================================================

def run_hyperparam_sweep(seed: int = SEED):
    """
    Sweep learning rate for DQN and PPO.
    Returns dict of {algo_lr: episode_rewards}.
    """
    sweep_results = {}

    # DQN learning rate sweep
    dqn_lrs = [5e-5, 1e-4, 5e-4]
    print("\n--- DQN Learning Rate Sweep ---")
    for lr in dqn_lrs:
        label = f"DQN_lr{lr}"
        _, rewards, wall_time = train_agent(
            "DQN", seed=seed, total_timesteps=100_000,
            learning_rate=lr,
        )
        sweep_results[label] = {
            "rewards": rewards,
            "lr": lr,
            "wall_time": wall_time,
            "mean_reward_last_50": float(np.mean(rewards[-50:])) if len(rewards) >= 50 else float(np.mean(rewards)),
        }

    # PPO learning rate sweep
    ppo_lrs = [1e-4, 3e-4, 1e-3]
    print("\n--- PPO Learning Rate Sweep ---")
    for lr in ppo_lrs:
        label = f"PPO_lr{lr}"
        _, rewards, wall_time = train_agent(
            "PPO", seed=seed, total_timesteps=100_000,
            learning_rate=lr,
        )
        sweep_results[label] = {
            "rewards": rewards,
            "lr": lr,
            "wall_time": wall_time,
            "mean_reward_last_50": float(np.mean(rewards[-50:])) if len(rewards) >= 50 else float(np.mean(rewards)),
        }

    return sweep_results


# =====================================================================
# Main Pipeline
# =====================================================================

def run_full_pipeline(skip_training=False, sweep_only=False):
    """Run the complete experiment pipeline."""

    all_results = {}

    if sweep_only:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER SWEEP ONLY")
        print("=" * 60)
        all_results["sweep"] = run_hyperparam_sweep()
        _save_results(all_results)
        return all_results

    # ------------------------------------------------------------------
    # Step 1: Train DQN and PPO
    # ------------------------------------------------------------------
    training_meta = {}

    if not skip_training:
        print("\n" + "=" * 60)
        print("STEP 1: TRAINING")
        print("=" * 60)

        _, dqn_rewards, dqn_time = train_agent("DQN", seed=SEED)
        training_meta["DQN"] = {
            "episode_rewards": dqn_rewards,
            "wall_time": dqn_time,
            "total_timesteps": TOTAL_TIMESTEPS,
        }

        _, ppo_rewards, ppo_time = train_agent("PPO", seed=SEED)
        training_meta["PPO"] = {
            "episode_rewards": ppo_rewards,
            "wall_time": ppo_time,
            "total_timesteps": TOTAL_TIMESTEPS,
        }
    else:
        print("\n  Skipping training (--eval-only). Loading saved rewards...")
        for algo in ["DQN", "PPO"]:
            rpath = RESULTS_DIR / f"{algo.lower()}_episode_rewards.npy"
            if rpath.exists():
                training_meta[algo] = {
                    "episode_rewards": np.load(str(rpath)).tolist(),
                    "wall_time": 0,
                    "total_timesteps": TOTAL_TIMESTEPS,
                }

    all_results["training"] = training_meta

    # ------------------------------------------------------------------
    # Step 2: Evaluate all 4 agents
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: EVALUATION (100 episodes each)")
    print("=" * 60)

    # Random
    random_agent = RandomAgent(n_actions=9, seed=SEED)
    all_results["Random"] = evaluate_agent(
        "Random",
        lambda obs, env: random_agent.act(env),
    )

    # Greedy
    greedy_agent = GreedyBaselineAgent()
    all_results["Greedy"] = evaluate_agent(
        "Greedy",
        lambda obs, env: greedy_agent.act(env),
    )

    # DQN
    dqn_path = RESULTS_DIR / "dqn_best" / "best_model.zip"
    if dqn_path.exists():
        dqn_model = DQN.load(str(dqn_path.with_suffix("")))
        all_results["DQN"] = evaluate_agent(
            "DQN",
            lambda obs, env: int(dqn_model.predict(obs, deterministic=True)[0]),
        )
    else:
        print("  DQN model not found — skipping evaluation")

    # PPO
    ppo_path = RESULTS_DIR / "ppo_best" / "best_model.zip"
    if ppo_path.exists():
        ppo_model = PPO.load(str(ppo_path.with_suffix("")))
        all_results["PPO"] = evaluate_agent(
            "PPO",
            lambda obs, env: int(ppo_model.predict(obs, deterministic=True)[0]),
        )
    else:
        print("  PPO model not found — skipping evaluation")

    # ------------------------------------------------------------------
    # Step 3: Hyperparameter sweep
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: HYPERPARAMETER SWEEP")
    print("=" * 60)
    all_results["sweep"] = run_hyperparam_sweep()

    # ------------------------------------------------------------------
    # Save everything
    # ------------------------------------------------------------------
    _save_results(all_results)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Agent':<10} {'Mean Reward':>14} {'Success Rate':>14} {'Avg Fire Left':>14}")
    print("-" * 55)
    for name in ["Random", "Greedy", "DQN", "PPO"]:
        if name in all_results and "mean_reward" in all_results[name]:
            r = all_results[name]
            print(f"{name:<10} {r['mean_reward']:>14.3f} {r['success_rate']:>13.1f}% "
                  f"{r['mean_fire_remaining']:>14.2f}")

    return all_results


def _save_results(all_results):
    path = RESULTS_DIR / "experiment_results.pkl"
    with open(path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\n  All results saved to {path}")


# =====================================================================
# Entry Point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full experiment pipeline")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, evaluate existing models")
    parser.add_argument("--sweep-only", action="store_true",
                        help="Only run hyperparameter sweep")
    args = parser.parse_args()

    run_full_pipeline(
        skip_training=args.eval_only,
        sweep_only=args.sweep_only,
    )
