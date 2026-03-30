"""
generate_plots.py
Generate all publication-quality visualizations from experiment results.

Usage:
    python experiments/generate_plots.py

Reads: results/experiment_results.pkl
Outputs: 8+ PNG files in results/
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"

# =====================================================================
# Style Configuration — Gartner-inspired
# =====================================================================

COLORS = {
    "Random": "#95a5a6",   # gray
    "Greedy": "#27ae60",   # green
    "DQN":    "#2980b9",   # blue
    "PPO":    "#e67e22",   # orange
}

AGENT_ORDER = ["Random", "Greedy", "DQN", "PPO"]

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "grid.color": "#2a2a4a",
    "legend.facecolor": "#16213e",
    "legend.edgecolor": "#e0e0e0",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})


def load_results():
    path = RESULTS_DIR / "experiment_results.pkl"
    if not path.exists():
        print(f"ERROR: {path} not found. Run experiments first:")
        print("  python experiments/run_experiments.py")
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)


# =====================================================================
# Plot 1: Training Curves (DQN vs PPO)
# =====================================================================

def plot_training_curves(results):
    fig, ax = plt.subplots(figsize=(10, 5))

    training = results.get("training", {})
    for algo in ["DQN", "PPO"]:
        if algo not in training:
            continue
        rewards = np.array(training[algo]["episode_rewards"])
        window = max(1, len(rewards) // 20)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        color = COLORS[algo]
        ax.plot(rewards, alpha=0.12, color=color, linewidth=0.5)
        ax.plot(smoothed, label=f"{algo} (smoothed)", color=color, linewidth=2.5)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Curves — DQN vs PPO")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = RESULTS_DIR / "01_training_curves.png"
    plt.savefig(path, dpi=200)
    print(f"  Saved: {path}")
    plt.close()


# =====================================================================
# Plot 2: Success Rate Bar Chart
# =====================================================================

def plot_success_rates(results):
    agents = [a for a in AGENT_ORDER if a in results and "success_rate" in results[a]]
    values = [results[a]["success_rate"] for a in agents]
    colors = [COLORS[a] for a in agents]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(agents, values, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
    ax.bar_label(bars, fmt="%.1f%%", padding=5, fontsize=12, fontweight="bold")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Fire Suppression Success Rate")
    ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 100)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = RESULTS_DIR / "02_success_rate.png"
    plt.savefig(path, dpi=200)
    print(f"  Saved: {path}")
    plt.close()


# =====================================================================
# Plot 3: Mean Reward Bar Chart
# =====================================================================

def plot_mean_rewards(results):
    agents = [a for a in AGENT_ORDER if a in results and "mean_reward" in results[a]]
    values = [results[a]["mean_reward"] for a in agents]
    stds = [results[a]["std_reward"] for a in agents]
    colors = [COLORS[a] for a in agents]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(agents, values, yerr=stds, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.6, capsize=5, error_kw={"linewidth": 1.5})
    ax.bar_label(bars, fmt="%.2f", padding=8, fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Mean Reward Comparison (100 Episodes)")
    ymin = min(0, min(values) * 1.2)
    ymax = max(values) * 1.3 if max(values) > 0 else 1
    ax.set_ylim(ymin, ymax)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = RESULTS_DIR / "03_mean_reward.png"
    plt.savefig(path, dpi=200)
    print(f"  Saved: {path}")
    plt.close()


# =====================================================================
# Plot 4: Position Heatmaps (2x2 grid)
# =====================================================================

def plot_position_heatmaps(results):
    agents = [a for a in AGENT_ORDER if a in results and "position_counts" in results[a]]
    if len(agents) < 2:
        print("  Skipping heatmaps — need at least 2 agents")
        return

    n = len(agents)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for idx, agent in enumerate(agents):
        ax = axes[idx]
        counts = results[agent]["position_counts"]
        # Normalize to percentage
        total = counts.sum() if counts.sum() > 0 else 1
        pct = counts / total * 100

        im = ax.imshow(pct, cmap="YlOrRd", interpolation="nearest", vmin=0)
        for r in range(counts.shape[0]):
            for c in range(counts.shape[1]):
                ax.text(c, r, f"{pct[r, c]:.1f}%", ha="center", va="center",
                        color="black" if pct[r, c] < pct.max() * 0.6 else "white",
                        fontsize=9, fontweight="bold")
        ax.set_title(agent, fontsize=13)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))

    fig.suptitle("Agent Position Frequency (% of total steps)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = RESULTS_DIR / "04_position_heatmaps.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# =====================================================================
# Plot 5: Fire Remaining Over Time
# =====================================================================

def plot_fire_over_time(results):
    fig, ax = plt.subplots(figsize=(10, 5))

    for agent in AGENT_ORDER:
        if agent not in results or "fire_over_time" not in results[agent]:
            continue

        traces = results[agent]["fire_over_time"]
        # Pad traces to same length and average
        max_len = MAX_STEPS = 200
        padded = []
        for trace in traces:
            if len(trace) < max_len:
                # Pad with last value
                trace = trace + [trace[-1]] * (max_len - len(trace))
            padded.append(trace[:max_len])

        mean_trace = np.mean(padded, axis=0)
        std_trace = np.std(padded, axis=0)

        color = COLORS[agent]
        ax.plot(mean_trace, label=agent, color=color, linewidth=2)
        ax.fill_between(range(len(mean_trace)),
                        mean_trace - std_trace, mean_trace + std_trace,
                        color=color, alpha=0.15)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Fire Cells Remaining")
    ax.set_title("Fire Suppression Speed — Average Fire Cells Over Time")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = RESULTS_DIR / "05_fire_over_time.png"
    plt.savefig(path, dpi=200)
    print(f"  Saved: {path}")
    plt.close()


# =====================================================================
# Plot 6: Training Wall-Clock Time
# =====================================================================

def plot_training_time(results):
    training = results.get("training", {})
    if not training:
        print("  Skipping training time plot — no training data")
        return

    agents = [a for a in ["DQN", "PPO"] if a in training]
    times = [training[a]["wall_time"] for a in agents]
    colors = [COLORS[a] for a in agents]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(agents, times, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
    ax.bar_label(bars, fmt="%.0fs", padding=5, fontsize=12, fontweight="bold")
    ax.set_ylabel("Wall-Clock Time (seconds)")
    ax.set_title("Training Time (200K Timesteps)")
    ax.set_ylim(0, max(times) * 1.3 if times else 100)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = RESULTS_DIR / "06_training_time.png"
    plt.savefig(path, dpi=200)
    print(f"  Saved: {path}")
    plt.close()


# =====================================================================
# Plot 7: Hyperparameter Sensitivity
# =====================================================================

def plot_hyperparam_sweep(results):
    sweep = results.get("sweep", {})
    if not sweep:
        print("  Skipping hyperparameter plot — no sweep data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, algo in enumerate(["DQN", "PPO"]):
        ax = axes[idx]
        entries = {k: v for k, v in sweep.items() if k.startswith(algo)}
        if not entries:
            continue

        lrs = [v["lr"] for v in entries.values()]
        mean_rewards = [v["mean_reward_last_50"] for v in entries.values()]

        # Sort by learning rate
        sorted_pairs = sorted(zip(lrs, mean_rewards))
        lrs, mean_rewards = zip(*sorted_pairs)

        color = COLORS[algo]
        ax.plot(lrs, mean_rewards, "o-", color=color, linewidth=2, markersize=10,
                markeredgecolor="white", markeredgewidth=1.5)

        for lr, mr in zip(lrs, mean_rewards):
            ax.annotate(f"{mr:.2f}", (lr, mr), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=10, fontweight="bold")

        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Mean Reward (Last 50 Episodes)")
        ax.set_title(f"{algo} — Learning Rate Sensitivity")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    path = RESULTS_DIR / "07_hyperparam_sweep.png"
    plt.savefig(path, dpi=200)
    print(f"  Saved: {path}")
    plt.close()


# =====================================================================
# Plot 8: Grid Snapshot Sequence
# =====================================================================

def plot_grid_snapshots(results):
    """Generate grid snapshots from a greedy episode for visual storytelling."""
    from environment.forest_fire_wrapper import ForestFireWrapper
    from agents.greedy_baseline import GreedyBaselineAgent

    env = ForestFireWrapper(max_steps=200, seed=42)
    agent = GreedyBaselineAgent()

    obs, _ = env.reset(seed=42)
    snapshots = []  # (step, grid, agent_pos)
    target_steps = [0, 50, 100, 150]

    snapshots.append((0, env.grid.copy(), env.agent_pos))

    for step in range(1, 201):
        action = agent.act(env)
        obs, reward, terminated, truncated, info = env.step(action)
        if step in target_steps:
            snapshots.append((step, env.grid.copy(), env.agent_pos))
        if terminated or truncated:
            if len(snapshots) < 4:
                snapshots.append((step, env.grid.copy(), env.agent_pos))
            break

    env.close()

    # Pad to 4 if episode ended early
    while len(snapshots) < 4:
        snapshots.append(snapshots[-1])

    snapshots = snapshots[:4]

    color_map = {
        0: np.array([0.7,  0.7,  0.7]),   # empty — light gray
        1: np.array([0.13, 0.55, 0.13]),   # tree  — green
        2: np.array([1.0,  0.27, 0.0]),    # fire  — orange-red
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for idx, (step, grid, pos) in enumerate(snapshots):
        ax = axes[idx]
        rgb = np.array([[color_map[int(v)] for v in row] for row in grid])
        ax.imshow(rgb, interpolation="nearest")

        ar, ac = pos
        ax.plot(ac, ar, "w^", markersize=14, markeredgecolor="black", markeredgewidth=1.5)

        fire_count = int(np.sum(grid == 2))
        tree_count = int(np.sum(grid == 1))
        ax.set_title(f"Step {step}\nFire: {fire_count} | Trees: {tree_count}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Greedy Agent — Episode Progression", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = RESULTS_DIR / "08_grid_snapshots.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# =====================================================================
# Main
# =====================================================================

def generate_all_plots():
    print("Loading experiment results...")
    results = load_results()

    print("\nGenerating plots:\n")
    plot_training_curves(results)
    plot_success_rates(results)
    plot_mean_rewards(results)
    plot_position_heatmaps(results)
    plot_fire_over_time(results)
    plot_training_time(results)
    plot_hyperparam_sweep(results)
    plot_grid_snapshots(results)

    print(f"\nAll plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    generate_all_plots()
