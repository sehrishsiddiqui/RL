"""
train_optimized.py
====================
Full pipeline: train DQN v3 + RecurrentPPO v3, evaluate all agents, generate comparison plots.

Usage:
    python training/train_optimized.py                     # train both + compare
    python training/train_optimized.py --dqn-only          # train DQN only
    python training/train_optimized.py --ppo-only          # train PPO only
    python training/train_optimized.py --eval-only         # skip training, just eval + plot
    python training/train_optimized.py --timesteps 500000  # quick test run
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from environment.forest_fire_wrapper import ForestFireWrapper
from agents.random_agent import RandomAgent
from agents.greedy_baseline import GreedyBaselineAgent as GreedyAgent

SAVE_DIR = Path(__file__).parent.parent / "results"
SAVE_DIR.mkdir(exist_ok=True)

N_EVAL_EPISODES = 100
EVAL_SEED = 999


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_sb3_model(model, n_episodes: int = N_EVAL_EPISODES, seed: int = EVAL_SEED,
                        is_recurrent: bool = False):
    """Evaluate an SB3 model. Returns (mean_reward, std_reward, success_rate)."""
    env = Monitor(ForestFireWrapper(max_steps=300, seed=seed))
    rewards = []
    successes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        lstm_states = None
        ep_start = np.ones((1,), dtype=bool)

        while not done:
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs, state=lstm_states, episode_start=ep_start, deterministic=True
                )
                ep_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)
        successes.append(float(info.get("success", False)))

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards)), float(np.mean(successes))


def evaluate_random(n_episodes: int = N_EVAL_EPISODES, seed: int = EVAL_SEED):
    """Evaluate random agent."""
    env = ForestFireWrapper(max_steps=300, seed=seed)
    agent = RandomAgent(n_actions=9, seed=seed)
    rewards, successes = [], []

    for _ in range(n_episodes):
        env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.act(env)
            _, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        successes.append(float(info.get("success", False)))

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards)), float(np.mean(successes))


def evaluate_greedy(n_episodes: int = N_EVAL_EPISODES, seed: int = EVAL_SEED):
    """Evaluate greedy BFS agent."""
    env = ForestFireWrapper(max_steps=300, seed=seed)
    agent = GreedyAgent()
    rewards, successes = [], []

    for _ in range(n_episodes):
        env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.act(env)
            _, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        successes.append(float(info.get("success", False)))

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards)), float(np.mean(successes))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
GRID_COLOR = "#30363d"
TEXT_COLOR = "#e6edf3"
COLORS = {
    "Random":        "#6e7681",
    "Greedy":        "#58a6ff",
    "DQN v3":        "#3fb950",
    "RecurrentPPO":  "#f78166",
}


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)


def plot_training_curves(dqn_rewards, ppo_rewards, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Training Curves — v3 Optimized", color=TEXT_COLOR, fontsize=14, y=1.02)

    for ax, rewards, label, color in zip(
        axes,
        [dqn_rewards, ppo_rewards],
        ["DQN v3", "RecurrentPPO v3"],
        [COLORS["DQN v3"], COLORS["RecurrentPPO"]],
    ):
        _style_ax(ax)
        if len(rewards) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", color=TEXT_COLOR)
            ax.set_title(label)
            continue

        rewards_arr = np.array(rewards)
        episodes = np.arange(len(rewards_arr))

        # Raw (faint)
        ax.plot(episodes, rewards_arr, color=color, alpha=0.2, linewidth=0.5)

        # Smoothed (50-ep moving average)
        if len(rewards_arr) >= 50:
            smooth = np.convolve(rewards_arr, np.ones(50) / 50, mode="valid")
            ax.plot(np.arange(len(smooth)) + 49, smooth, color=color, linewidth=2, label=label)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.set_title(f"{label} — {len(rewards_arr)} episodes")
        ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {save_path}")


def plot_comparison(results: dict, save_path):
    """Bar charts: mean reward and success rate for all 4 agents."""
    agents = list(results.keys())
    mean_rewards = [results[a]["mean_reward"] for a in agents]
    std_rewards  = [results[a]["std_reward"] for a in agents]
    success_rates = [results[a]["success_rate"] * 100 for a in agents]
    colors = [COLORS.get(a, "#aaa") for a in agents]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f"Agent Comparison — {N_EVAL_EPISODES} eval episodes", color=TEXT_COLOR, fontsize=13)

    for ax in (ax1, ax2):
        _style_ax(ax)

    # Mean reward
    bars = ax1.bar(agents, mean_rewards, yerr=std_rewards, color=colors,
                   capsize=5, error_kw=dict(ecolor=TEXT_COLOR, lw=1.5))
    for bar, val in zip(bars, mean_rewards):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
    ax1.set_ylabel("Mean Episode Reward")
    ax1.set_title("Mean Reward (±1 std)")
    ax1.tick_params(axis="x", rotation=15)

    # Success rate
    bars2 = ax2.bar(agents, success_rates, color=colors)
    for bar, val in zip(bars2, success_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title("Fire Suppression Success Rate")
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {save_path}")


def plot_reward_improvement(results: dict, save_path):
    """Show % improvement of each agent over random baseline."""
    random_mean = results["Random"]["mean_reward"]
    agents = [a for a in results if a != "Random"]
    improvements = [
        (results[a]["mean_reward"] - random_mean) / max(abs(random_mean), 1e-6) * 100
        for a in agents
    ]
    colors = [COLORS.get(a, "#aaa") for a in agents]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax)

    bars = ax.bar(agents, improvements, color=colors)
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (1 if val >= 0 else -3),
                f"{val:+.1f}%", ha="center", va="bottom", color=TEXT_COLOR, fontsize=10)

    ax.axhline(0, color=GRID_COLOR, linewidth=1)
    ax.set_ylabel("% Improvement over Random")
    ax.set_title("Reward Improvement vs. Random Baseline", color=TEXT_COLOR)
    ax.tick_params(axis="x", rotation=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Wildfire RL v3 — optimized training + eval")
    parser.add_argument("--dqn-only",  action="store_true", help="Train DQN only")
    parser.add_argument("--ppo-only",  action="store_true", help="Train PPO only")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, evaluate saved models")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total training timesteps")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    dqn_rewards, ppo_rewards = [], []

    # ---- Training ----
    if not args.eval_only:
        if not args.ppo_only:
            from training.train_dqn import train_dqn
            t0 = time.time()
            _, dqn_rewards = train_dqn(total_timesteps=args.timesteps, seed=args.seed)
            print(f"DQN training time: {(time.time()-t0)/60:.1f} min")

        if not args.dqn_only:
            from training.train_ppo import train_ppo
            t0 = time.time()
            _, ppo_rewards = train_ppo(total_timesteps=args.timesteps, seed=args.seed)
            print(f"RecurrentPPO training time: {(time.time()-t0)/60:.1f} min")

    # Load saved rewards if eval-only
    if args.eval_only or dqn_rewards == []:
        dqn_path = SAVE_DIR / "dqn_episode_rewards.npy"
        if dqn_path.exists():
            dqn_rewards = np.load(str(dqn_path)).tolist()
    if args.eval_only or ppo_rewards == []:
        ppo_path = SAVE_DIR / "ppo_episode_rewards.npy"
        if ppo_path.exists():
            ppo_rewards = np.load(str(ppo_path)).tolist()

    # ---- Evaluation ----
    print(f"\n=== Evaluating all agents ({N_EVAL_EPISODES} episodes each) ===")

    results = {}

    print("  Random agent...")
    m, s, sr = evaluate_random()
    results["Random"] = {"mean_reward": m, "std_reward": s, "success_rate": sr}
    print(f"    reward={m:.2f}±{s:.2f}  success={sr:.2%}")

    print("  Greedy agent...")
    m, s, sr = evaluate_greedy()
    results["Greedy"] = {"mean_reward": m, "std_reward": s, "success_rate": sr}
    print(f"    reward={m:.2f}±{s:.2f}  success={sr:.2%}")

    dqn_model_path = SAVE_DIR / "dqn_best" / "best_model.zip"
    if dqn_model_path.exists():
        print("  DQN v3...")
        model = DQN.load(str(dqn_model_path))
        m, s, sr = evaluate_sb3_model(model, is_recurrent=False)
        results["DQN v3"] = {"mean_reward": m, "std_reward": s, "success_rate": sr}
        print(f"    reward={m:.2f}±{s:.2f}  success={sr:.2%}")
    else:
        print("  DQN model not found, skipping.")

    ppo_model_path = SAVE_DIR / "ppo_best" / "best_model.zip"
    if ppo_model_path.exists():
        print("  RecurrentPPO v3...")
        model = RecurrentPPO.load(str(ppo_model_path))
        m, s, sr = evaluate_sb3_model(model, is_recurrent=True)
        results["RecurrentPPO"] = {"mean_reward": m, "std_reward": s, "success_rate": sr}
        print(f"    reward={m:.2f}±{s:.2f}  success={sr:.2%}")
    else:
        print("  PPO model not found, skipping.")

    # ---- Summary table ----
    print("\n" + "="*65)
    print(f"{'Agent':<18} {'Mean Reward':>14} {'Std':>10} {'Success %':>12}")
    print("-"*65)
    for agent, res in results.items():
        print(f"{agent:<18} {res['mean_reward']:>14.2f} {res['std_reward']:>10.2f} "
              f"{res['success_rate']*100:>11.1f}%")
    print("="*65)

    # ---- Plots ----
    print("\nGenerating plots...")
    if dqn_rewards or ppo_rewards:
        plot_training_curves(dqn_rewards, ppo_rewards,
                             SAVE_DIR / "v3_training_curves.png")

    if results:
        plot_comparison(results, SAVE_DIR / "v3_agent_comparison.png")
        plot_reward_improvement(results, SAVE_DIR / "v3_improvement_vs_random.png")

    print(f"\nAll results saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
