"""
visualization.py
Plotting utilities for training curves, heatmaps, and episode analysis.
Updated for ForestFireHelicopter5x5-v1 (no populated zones, 5x5 grid).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

AGENT_COLORS = {"DQN": "steelblue", "PPO": "darkorange", "Greedy": "green"}


def plot_training_curves(rewards_dict: dict, title: str = "Training Curves", save: bool = True):
    """
    Plot smoothed episode reward curves for multiple agents.

    Args:
        rewards_dict: {"DQN": [r1, r2, ...], "PPO": [...], "Greedy": [...]}
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for agent_name, rewards in rewards_dict.items():
        rewards = np.array(rewards)
        window = max(1, len(rewards) // 20)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        color = AGENT_COLORS.get(agent_name, "gray")
        ax.plot(smoothed, label=f"{agent_name} (smoothed)", color=color, linewidth=2)
        ax.plot(rewards, alpha=0.15, color=color, linewidth=0.5)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = RESULTS_DIR / "training_curves.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


def plot_agent_heatmap(position_counts: np.ndarray, title: str = "Agent Position Heatmap",
                       save: bool = True):
    """
    Heatmap of how often the helicopter visited each cell across episodes.

    Args:
        position_counts: (5, 5) array of visit counts
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(position_counts, cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Visit count")

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Label each cell with its count
    for r in range(position_counts.shape[0]):
        for c in range(position_counts.shape[1]):
            ax.text(c, r, f"{int(position_counts[r, c])}",
                    ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()

    if save:
        fname = title.lower().replace(" ", "_") + ".png"
        path = RESULTS_DIR / fname
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


def plot_comparison_bar(metrics: dict, metric_name: str = "Fire Suppression Success Rate (%)",
                        save: bool = True):
    """
    Bar chart comparing a metric across agents.

    Args:
        metrics: {"DQN": value, "PPO": value, "Greedy": value}
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    agents = list(metrics.keys())
    values = list(metrics.values())
    colors = [AGENT_COLORS.get(a, "gray") for a in agents]

    bars = ax.bar(agents, values, color=colors, edgecolor="black", alpha=0.85)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Agent Comparison — {metric_name}")
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        fname = metric_name.lower().replace(" ", "_").replace("(%)", "").strip() + ".png"
        path = RESULTS_DIR / fname
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


def render_grid_snapshot(grid: np.ndarray, agent_pos: tuple, step: int, save: bool = False):
    """
    Render a single timestep of the ForestFire grid using matplotlib.
    Grid values: 0=empty (gray), 1=tree (green), 2=fire (red).

    Args:
        grid:      (5, 5) numpy array
        agent_pos: (row, col) of helicopter
        step:      current timestep
    """
    color_map = {
        0: [0.7,  0.7,  0.7],   # empty — gray
        1: [0.13, 0.55, 0.13],  # tree  — green
        2: [1.0,  0.27, 0.0],   # fire  — orange-red
    }

    rgb = np.array([[color_map[int(v)] for v in row] for row in grid])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rgb, interpolation="nearest")

    ar, ac = agent_pos
    ax.plot(ac, ar, "w^", markersize=14, markeredgecolor="black", markeredgewidth=1.5,
            label="Helicopter")

    ax.set_title(f"Step {step} | Fire cells: {int(np.sum(grid == 2))}")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    if save:
        path = RESULTS_DIR / f"grid_step_{step:04d}.png"
        plt.savefig(path, dpi=150)
    plt.show()
