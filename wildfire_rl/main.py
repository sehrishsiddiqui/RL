"""
main.py
Entry point for the Wildfire RL project using ForestFireHelicopter5x5-v1.

Usage:
    python main.py --agent random            # run random baseline
    python main.py --agent greedy           # run greedy baseline
    python main.py --agent dqn --train      # train DQN from scratch
    python main.py --agent ppo --train      # train PPO from scratch
    python main.py --agent dqn --eval       # evaluate saved DQN model
    python main.py --agent ppo --eval       # evaluate saved PPO model
    python main.py --compare                # run all four agents and compare
    python main.py --experiment             # run full experiment pipeline
"""

import argparse
import sys
import numpy as np
from pathlib import Path

from environment.forest_fire_wrapper import ForestFireWrapper
from agents.greedy_baseline import GreedyBaselineAgent
from agents.random_agent import RandomAgent
from utils.visualization import plot_training_curves, plot_comparison_bar


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_random(n_episodes: int = 50, seed: int = 42) -> dict:
    """Run the random baseline and return episode stats."""
    env = ForestFireWrapper(max_steps=200, seed=seed)
    agent = RandomAgent(n_actions=9, seed=seed)

    rewards, successes, steps, fire_remaining = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        info = {}

        while not done:
            action = agent.act(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        successes.append(1 if info.get("success", False) else 0)
        steps.append(info.get("step", 0))
        fire_remaining.append(info.get("fire_remaining", -1))

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes} | Reward: {total_reward:.2f} | "
                  f"Success: {bool(successes[-1])} | Fire left: {fire_remaining[-1]}")

    env.close()

    print(f"\nRandom Baseline Results ({n_episodes} episodes):")
    print(f"  Mean reward:       {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  Success rate:      {np.mean(successes)*100:.1f}%")
    print(f"  Mean steps:        {np.mean(steps):.1f}")
    print(f"  Mean fire left:    {np.mean(fire_remaining):.2f} cells")

    return {
        "rewards": rewards,
        "success_rate": np.mean(successes) * 100,
        "mean_reward": np.mean(rewards),
        "mean_steps": np.mean(steps),
    }


def run_greedy(n_episodes: int = 50, seed: int = 42) -> dict:
    """Run the greedy baseline and return episode stats."""
    env = ForestFireWrapper(max_steps=200, seed=seed)
    agent = GreedyBaselineAgent()

    rewards, successes, steps, fire_remaining = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        info = {}

        while not done:
            action = agent.act(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        successes.append(1 if info.get("success", False) else 0)
        steps.append(info.get("step", 0))
        fire_remaining.append(info.get("fire_remaining", -1))

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes} | Reward: {total_reward:.2f} | "
                  f"Success: {bool(successes[-1])} | Fire left: {fire_remaining[-1]}")

    env.close()

    print(f"\nGreedy Baseline Results ({n_episodes} episodes):")
    print(f"  Mean reward:       {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  Success rate:      {np.mean(successes)*100:.1f}%")
    print(f"  Mean steps:        {np.mean(steps):.1f}")
    print(f"  Mean fire left:    {np.mean(fire_remaining):.2f} cells")

    return {
        "rewards": rewards,
        "success_rate": np.mean(successes) * 100,
        "mean_reward": np.mean(rewards),
        "mean_steps": np.mean(steps),
    }


def eval_model(agent_name: str, n_episodes: int = 50, seed: int = 42) -> dict:
    """Evaluate a saved SB3 model."""
    from stable_baselines3 import DQN, PPO

    model_path = RESULTS_DIR / f"{agent_name}_best" / "best_model"
    if not model_path.with_suffix(".zip").exists():
        print(f"No saved model found at {model_path}.zip")
        print(f"Run: python main.py --agent {agent_name} --train")
        sys.exit(1)

    ModelClass = DQN if agent_name == "dqn" else PPO
    model = ModelClass.load(str(model_path))

    env = ForestFireWrapper(max_steps=200, seed=seed)
    rewards, successes, steps, fire_remaining = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        successes.append(1 if info.get("success", False) else 0)
        steps.append(info.get("step", 0))
        fire_remaining.append(info.get("fire_remaining", -1))

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes} | Reward: {total_reward:.2f} | "
                  f"Success: {bool(successes[-1])} | Fire left: {fire_remaining[-1]}")

    env.close()

    print(f"\n{agent_name.upper()} Evaluation ({n_episodes} episodes):")
    print(f"  Mean reward:       {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  Success rate:      {np.mean(successes)*100:.1f}%")
    print(f"  Mean steps:        {np.mean(steps):.1f}")
    print(f"  Mean fire left:    {np.mean(fire_remaining):.2f} cells")

    return {
        "rewards": rewards,
        "success_rate": np.mean(successes) * 100,
        "mean_reward": np.mean(rewards),
        "mean_steps": np.mean(steps),
    }


def compare_all(n_episodes: int = 50, seed: int = 42):
    """Run all four agents and produce comparison plots."""
    print("=" * 55)
    print("AGENT COMPARISON — ForestFireHelicopter5x5-v1")
    print("=" * 55)

    results = {}

    print("\n--- Random Baseline ---")
    results["Random"] = run_random(n_episodes=n_episodes, seed=seed)

    print("\n--- Greedy Baseline ---")
    results["Greedy"] = run_greedy(n_episodes=n_episodes, seed=seed)

    print("\n--- DQN ---")
    results["DQN"] = eval_model("dqn", n_episodes=n_episodes, seed=seed)

    print("\n--- PPO ---")
    results["PPO"] = eval_model("ppo", n_episodes=n_episodes, seed=seed)

    print("\n" + "=" * 55)
    print(f"{'Agent':<10} {'Mean Reward':>14} {'Success Rate':>14} {'Mean Steps':>12}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<10} {r['mean_reward']:>14.3f} {r['success_rate']:>13.1f}% {r['mean_steps']:>12.1f}")

    plot_comparison_bar(
        {name: r["success_rate"] for name, r in results.items()},
        metric_name="Fire Suppression Success Rate (%)",
    )
    plot_comparison_bar(
        {name: r["mean_reward"] for name, r in results.items()},
        metric_name="Mean Episode Reward",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ForestFire Helicopter RL")
    parser.add_argument("--agent", choices=["random", "greedy", "dqn", "ppo"], default="greedy")
    parser.add_argument("--train",      action="store_true", help="Train the agent from scratch")
    parser.add_argument("--eval",       action="store_true", help="Evaluate a saved model")
    parser.add_argument("--compare",    action="store_true", help="Compare all agents")
    parser.add_argument("--experiment", action="store_true", help="Run full experiment pipeline")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    if args.experiment:
        from experiments.run_experiments import run_full_pipeline
        run_full_pipeline()

    elif args.compare:
        compare_all(n_episodes=args.episodes, seed=args.seed)

    elif args.agent == "random":
        run_random(n_episodes=args.episodes, seed=args.seed)

    elif args.agent == "greedy":
        run_greedy(n_episodes=args.episodes, seed=args.seed)

    elif args.train:
        if args.agent == "dqn":
            from training.train_dqn import train_dqn
            train_dqn(seed=args.seed)
        elif args.agent == "ppo":
            from training.train_ppo import train_ppo
            train_ppo(seed=args.seed)

    elif args.eval:
        eval_model(args.agent, n_episodes=args.episodes, seed=args.seed)

    else:
        parser.print_help()
