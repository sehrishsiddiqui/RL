"""
rigorous_eval.py
Rigorous evaluation framework that addresses data leakage, overfitting, and
ensures fair cross-algorithm comparison.

Key principles:
  - Eval seeds are COMPLETELY disjoint from training seeds
  - Tests both WITH and WITHOUT early termination
  - 500 episodes per agent for statistical significance
  - Confidence intervals on all metrics
  - Same episodes (same seeds) across all agents for fair comparison
  - Tracks per-step behavior, not just final outcome

Usage:
    python experiments/rigorous_eval.py                 # full eval
    python experiments/rigorous_eval.py --quick          # 100 episodes (faster)
    python experiments/rigorous_eval.py --train-first    # train then eval
"""

import sys
import time
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.forest_fire_wrapper import ForestFireWrapper
from agents.random_agent import RandomAgent
from agents.greedy_baseline import GreedyBaselineAgent

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# =====================================================================
# SEED PROTOCOL — completely disjoint ranges
# =====================================================================
# Training uses seeds:       0 - 999
# Eval callback uses seeds:  1000 - 1999
# Final evaluation uses:     10000 - 19999  (no overlap possible)
TRAIN_SEED_START = 0
EVAL_SEED_START = 10_000


# =====================================================================
# Evaluation Config
# =====================================================================
@dataclass
class EvalConfig:
    n_episodes: int = 500
    max_steps: int = 200
    seed_start: int = EVAL_SEED_START
    early_termination: bool = True


@dataclass
class EpisodeResult:
    seed: int = 0
    total_reward: float = 0.0
    success: bool = False
    steps: int = 0
    fire_remaining: int = 0
    trees_remaining: int = 0
    fires_extinguished: int = 0  # total hits across episode
    time_to_first_clear: int = -1  # step when fire first hit 0, -1 if never
    fire_trace: list = field(default_factory=list)


# =====================================================================
# Core evaluator — runs one agent across all episodes
# =====================================================================
def evaluate_agent(agent_name: str, get_action_fn, config: EvalConfig,
                   needs_env: bool = False) -> dict:
    """
    Evaluate a single agent rigorously.

    Args:
        agent_name: label
        get_action_fn: callable(obs, env) -> action
        config: EvalConfig
        needs_env: if True, env is passed to get_action_fn (for greedy)
    """
    episodes = []

    for ep_idx in range(config.n_episodes):
        seed = config.seed_start + ep_idx
        env = ForestFireWrapper(max_steps=config.max_steps, seed=seed)

        # Temporarily disable early termination if needed
        if not config.early_termination:
            env._disable_early_term = True

        obs, _ = env.reset(seed=seed)
        result = EpisodeResult(seed=seed)
        done = False
        first_clear_found = False

        while not done:
            if needs_env:
                action = get_action_fn(obs, env)
            else:
                action = get_action_fn(obs, env)

            obs, reward, terminated, truncated, info = env.step(action)

            # Override termination if we disabled early term
            if not config.early_termination and info.get("success", False):
                terminated = False  # force continue

            result.total_reward += reward
            result.steps = info.get("step", result.steps)
            result.fire_trace.append(info.get("fire_remaining", 0))

            if info.get("hit", False):
                result.fires_extinguished += 1

            if info.get("fire_remaining", 1) == 0 and not first_clear_found:
                result.time_to_first_clear = info.get("step", -1)
                first_clear_found = True

            done = terminated or truncated

        result.success = info.get("success", False)
        result.fire_remaining = info.get("fire_remaining", -1)
        result.trees_remaining = info.get("trees_remaining", 0)
        episodes.append(result)
        env.close()

    return _summarize(agent_name, episodes, config)


def _summarize(agent_name: str, episodes: list, config: EvalConfig) -> dict:
    """Compute comprehensive statistics from episode results."""
    n = len(episodes)
    rewards = np.array([e.total_reward for e in episodes])
    successes = np.array([1.0 if e.success else 0.0 for e in episodes])
    steps = np.array([e.steps for e in episodes])
    fire_remaining = np.array([e.fire_remaining for e in episodes])
    fires_extinguished = np.array([e.fires_extinguished for e in episodes])
    clear_times = [e.time_to_first_clear for e in episodes if e.time_to_first_clear > 0]

    # 95% confidence interval
    def ci95(arr):
        mean = np.mean(arr)
        se = np.std(arr, ddof=1) / np.sqrt(len(arr))
        return float(mean - 1.96 * se), float(mean + 1.96 * se)

    summary = {
        "agent": agent_name,
        "n_episodes": n,
        "early_termination": config.early_termination,
        "eval_seed_range": f"{config.seed_start}-{config.seed_start + n - 1}",

        # Core metrics
        "success_rate": float(np.mean(successes) * 100),
        "success_rate_ci95": [x * 100 for x in ci95(successes)],
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "reward_ci95": list(ci95(rewards)),
        "median_reward": float(np.median(rewards)),

        # Episode length
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "min_steps": int(np.min(steps)),
        "max_steps": int(np.max(steps)),

        # Fire suppression quality
        "mean_fire_remaining": float(np.mean(fire_remaining)),
        "mean_fires_extinguished": float(np.mean(fires_extinguished)),
        "mean_time_to_clear": float(np.mean(clear_times)) if clear_times else -1,
        "pct_ever_cleared": float(len(clear_times) / n * 100),

        # Distributional stats
        "reward_quartiles": [float(np.percentile(rewards, q)) for q in [25, 50, 75]],
        "success_count": int(np.sum(successes)),
    }

    return summary


# =====================================================================
# Cross-algorithm comparison
# =====================================================================
def compare_agents(results: list) -> str:
    """Generate a formatted comparison table and analysis."""
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("  RIGOROUS CROSS-ALGORITHM COMPARISON")
    lines.append("=" * 90)

    mode = "WITH early termination" if results[0]["early_termination"] else "WITHOUT early termination"
    lines.append(f"  Mode: {mode}")
    lines.append(f"  Episodes: {results[0]['n_episodes']}")
    lines.append(f"  Eval seeds: {results[0]['eval_seed_range']}")
    lines.append(f"  (Training seeds: 0-999 — ZERO overlap)")
    lines.append("")

    # Header
    header = f"{'Agent':<10} {'Success%':>10} {'95% CI':>18} {'Mean Rew':>10} {'Std Rew':>10} {'Avg Steps':>10} {'Fire Left':>10} {'Hits':>8} {'Clear Time':>11}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        ci_lo, ci_hi = r["success_rate_ci95"]
        clear_t = f"{r['mean_time_to_clear']:.1f}" if r['mean_time_to_clear'] > 0 else "N/A"
        line = (f"{r['agent']:<10} {r['success_rate']:>9.1f}% "
                f"[{ci_lo:>5.1f}%, {ci_hi:>5.1f}%] "
                f"{r['mean_reward']:>10.2f} {r['std_reward']:>10.2f} "
                f"{r['mean_steps']:>10.1f} {r['mean_fire_remaining']:>10.2f} "
                f"{r['mean_fires_extinguished']:>8.1f} {clear_t:>11}")
        lines.append(line)

    lines.append("")

    # Statistical comparison
    if len(results) >= 4:
        random_r = next(r for r in results if r["agent"] == "Random")
        greedy_r = next(r for r in results if r["agent"] == "Greedy")
        dqn_r = next((r for r in results if r["agent"] == "DQN"), None)
        ppo_r = next((r for r in results if r["agent"] == "PPO"), None)

        lines.append("  KEY FINDINGS:")
        lines.append(f"  - Greedy vs Random:  +{greedy_r['success_rate'] - random_r['success_rate']:.1f}% success, "
                      f"+{greedy_r['mean_reward'] - random_r['mean_reward']:.1f} reward")
        if dqn_r:
            lines.append(f"  - DQN vs Random:     +{dqn_r['success_rate'] - random_r['success_rate']:.1f}% success, "
                          f"+{dqn_r['mean_reward'] - random_r['mean_reward']:.1f} reward")
            lines.append(f"  - DQN vs Greedy:     {'+' if dqn_r['success_rate'] >= greedy_r['success_rate'] else ''}"
                          f"{dqn_r['success_rate'] - greedy_r['success_rate']:.1f}% success, "
                          f"{'+' if dqn_r['mean_reward'] >= greedy_r['mean_reward'] else ''}"
                          f"{dqn_r['mean_reward'] - greedy_r['mean_reward']:.1f} reward")
        if ppo_r:
            lines.append(f"  - PPO vs Random:     +{ppo_r['success_rate'] - random_r['success_rate']:.1f}% success, "
                          f"+{ppo_r['mean_reward'] - random_r['mean_reward']:.1f} reward")
            lines.append(f"  - PPO vs Greedy:     {'+' if ppo_r['success_rate'] >= greedy_r['success_rate'] else ''}"
                          f"{ppo_r['success_rate'] - greedy_r['success_rate']:.1f}% success, "
                          f"{'+' if ppo_r['mean_reward'] >= greedy_r['mean_reward'] else ''}"
                          f"{ppo_r['mean_reward'] - greedy_r['mean_reward']:.1f} reward")
        if dqn_r and ppo_r:
            lines.append(f"  - PPO vs DQN:        {'+' if ppo_r['success_rate'] >= dqn_r['success_rate'] else ''}"
                          f"{ppo_r['success_rate'] - dqn_r['success_rate']:.1f}% success, "
                          f"{'+' if ppo_r['mean_reward'] >= dqn_r['mean_reward'] else ''}"
                          f"{ppo_r['mean_reward'] - dqn_r['mean_reward']:.1f} reward")

        # Recommendation
        lines.append("")
        lines.append("  RECOMMENDATION:")
        all_agents = [r for r in results if r["agent"] in ("DQN", "PPO", "Greedy")]
        best = max(all_agents, key=lambda r: (r["success_rate"], r["mean_reward"]))
        lines.append(f"  Best overall: {best['agent']} ({best['success_rate']:.1f}% success, {best['mean_reward']:.1f} reward)")

        if dqn_r and ppo_r:
            if ppo_r["success_rate"] > dqn_r["success_rate"] + 2:
                lines.append("  PPO is the stronger RL agent — higher success rate with better stability.")
            elif dqn_r["success_rate"] > ppo_r["success_rate"] + 2:
                lines.append("  DQN is the stronger RL agent — higher success rate.")
            else:
                lines.append("  DQN and PPO perform similarly — PPO preferred if training time is not a constraint.")

    lines.append("=" * 90)
    return "\n".join(lines)


# =====================================================================
# Data leakage detector
# =====================================================================
def detect_leakage(results_with_term: list, results_without_term: list) -> str:
    """Compare with/without early termination to detect inflated metrics."""
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("  DATA LEAKAGE & OVERFITTING ANALYSIS")
    lines.append("=" * 90)

    for agent_name in ["Random", "Greedy", "DQN", "PPO"]:
        r_with = next((r for r in results_with_term if r["agent"] == agent_name), None)
        r_without = next((r for r in results_without_term if r["agent"] == agent_name), None)
        if not r_with or not r_without:
            continue

        delta_success = r_with["success_rate"] - r_without["success_rate"]
        lines.append(f"\n  {agent_name}:")
        lines.append(f"    With early term:    {r_with['success_rate']:>6.1f}% success, {r_with['mean_reward']:>8.1f} reward, {r_with['mean_steps']:>6.1f} avg steps")
        lines.append(f"    Without early term: {r_without['success_rate']:>6.1f}% success, {r_without['mean_reward']:>8.1f} reward, {r_without['mean_steps']:>6.1f} avg steps")
        lines.append(f"    Delta:              {delta_success:>+6.1f}% success")

        if delta_success > 30:
            lines.append(f"    WARNING: {agent_name} success drops {delta_success:.0f}% without early termination — "
                          "performance heavily depends on avoiding new fire spawns, not suppression strategy.")
        elif delta_success > 10:
            lines.append(f"    NOTE: Moderate dependence on early termination ({delta_success:.0f}% drop).")
        else:
            lines.append(f"    GOOD: Minimal dependence on early termination ({delta_success:.0f}% drop).")

    lines.append("\n  INTERPRETATION:")
    lines.append("  - Large gaps between with/without early termination suggest the agent")
    lines.append("    succeeds by clearing initial fires quickly, not by sustained suppression.")
    lines.append("  - The 'without early termination' numbers are the HONEST performance metric.")
    lines.append("  - For a fair comparison to the original results (which had no early termination),")
    lines.append("    use the 'without' numbers.")
    lines.append("=" * 90)
    return "\n".join(lines)


# =====================================================================
# Main pipeline
# =====================================================================
def load_trained_models():
    """Load trained DQN and PPO models if they exist."""
    models = {}
    try:
        from stable_baselines3 import DQN
        dqn_path = RESULTS_DIR / "dqn_best" / "best_model.zip"
        if dqn_path.exists():
            models["DQN"] = DQN.load(str(dqn_path.with_suffix("")))
            print(f"  Loaded DQN from {dqn_path}")
    except Exception as e:
        print(f"  Could not load DQN: {e}")

    try:
        from stable_baselines3 import PPO
        ppo_path = RESULTS_DIR / "ppo_best" / "best_model.zip"
        if ppo_path.exists():
            models["PPO"] = PPO.load(str(ppo_path.with_suffix("")))
            print(f"  Loaded PPO from {ppo_path}")
    except Exception as e:
        print(f"  Could not load PPO: {e}")

    return models


def train_fresh_models(timesteps: int = 2_000_000):
    """Train DQN and PPO from scratch with proper seed protocol."""
    import warnings
    warnings.filterwarnings("ignore")

    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    def linear_schedule(initial_lr):
        def schedule(progress_remaining):
            return progress_remaining * initial_lr
        return schedule

    class RewardLoggerCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self._current_rewards = None
        def _on_step(self):
            n = len(self.locals["rewards"])
            if self._current_rewards is None:
                self._current_rewards = [0.0] * n
            for i in range(n):
                self._current_rewards[i] += self.locals["rewards"][i]
                if self.locals["dones"][i]:
                    self.episode_rewards.append(self._current_rewards[i])
                    self._current_rewards[i] = 0.0
            return True

    TRAIN_SEED = 0  # training range: 0-999
    EVAL_CB_SEED = 1000  # eval callback range: 1000-1999

    models = {}

    # --- DQN ---
    print(f"\n  Training DQN for {timesteps:,} steps (seed range: {TRAIN_SEED}-999)...")
    env = Monitor(ForestFireWrapper(max_steps=200, seed=TRAIN_SEED))
    eval_env = Monitor(ForestFireWrapper(max_steps=200, seed=EVAL_CB_SEED))
    reward_cb = RewardLoggerCallback()
    eval_cb = EvalCallback(eval_env, best_model_save_path=str(RESULTS_DIR / "dqn_best"),
                           log_path=str(RESULTS_DIR / "dqn_eval_logs"),
                           eval_freq=10_000, n_eval_episodes=20, deterministic=True, verbose=0)
    dqn = DQN("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=linear_schedule(1e-4), buffer_size=200_000,
              learning_starts=5_000, batch_size=256, gamma=0.99,
              exploration_fraction=0.3, exploration_final_eps=0.02,
              target_update_interval=500, train_freq=4, verbose=0, seed=TRAIN_SEED)
    t0 = time.time()
    dqn.learn(total_timesteps=timesteps, callback=[reward_cb, eval_cb])
    dqn_time = time.time() - t0
    dqn.save(str(RESULTS_DIR / "dqn_final"))
    np.save(str(RESULTS_DIR / "dqn_episode_rewards.npy"), np.array(reward_cb.episode_rewards))
    models["DQN"] = dqn
    print(f"  DQN done in {dqn_time:.0f}s ({len(reward_cb.episode_rewards)} episodes)")
    env.close(); eval_env.close()

    # --- PPO ---
    print(f"\n  Training PPO for {timesteps:,} steps (seed range: {TRAIN_SEED}-999)...")
    env = DummyVecEnv([
        lambda i=i: Monitor(ForestFireWrapper(max_steps=200, seed=TRAIN_SEED + i))
        for i in range(4)
    ])
    eval_env = Monitor(ForestFireWrapper(max_steps=200, seed=EVAL_CB_SEED))
    reward_cb = RewardLoggerCallback()
    eval_cb = EvalCallback(eval_env, best_model_save_path=str(RESULTS_DIR / "ppo_best"),
                           log_path=str(RESULTS_DIR / "ppo_eval_logs"),
                           eval_freq=10_000, n_eval_episodes=20, deterministic=True, verbose=0)
    ppo = PPO("MlpPolicy", env,
              policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
              learning_rate=linear_schedule(3e-4), n_steps=2048, batch_size=128,
              n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
              ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5, verbose=0, seed=TRAIN_SEED)
    t0 = time.time()
    ppo.learn(total_timesteps=timesteps, callback=[reward_cb, eval_cb])
    ppo_time = time.time() - t0
    ppo.save(str(RESULTS_DIR / "ppo_final"))
    np.save(str(RESULTS_DIR / "ppo_episode_rewards.npy"), np.array(reward_cb.episode_rewards))
    models["PPO"] = ppo
    print(f"  PPO done in {ppo_time:.0f}s ({len(reward_cb.episode_rewards)} episodes)")
    env.close(); eval_env.close()

    return models, {"DQN": dqn_time, "PPO": ppo_time}


def run_full_eval(n_episodes: int = 500, train_first: bool = False,
                  train_timesteps: int = 2_000_000):
    """Run the complete rigorous evaluation."""
    import warnings
    warnings.filterwarnings("ignore")

    print("\n" + "=" * 90)
    print("  RIGOROUS EVALUATION FRAMEWORK")
    print("  Eval seeds: 10000+ (disjoint from training seeds 0-999)")
    print("=" * 90)

    # --- Load or train models ---
    train_times = {}
    if train_first:
        print("\n--- TRAINING ---")
        models, train_times = train_fresh_models(train_timesteps)
    else:
        print("\n--- LOADING MODELS ---")
        models = load_trained_models()

    # --- Setup agents ---
    random_agent = RandomAgent(n_actions=9, seed=99999)
    greedy_agent = GreedyBaselineAgent()

    def random_action(obs, env):
        return random_agent.act(env)

    def greedy_action(obs, env):
        return greedy_agent.act(env)

    agent_fns = {"Random": random_action, "Greedy": greedy_action}

    for name, model in models.items():
        def make_fn(m):
            def fn(obs, env):
                action, _ = m.predict(obs, deterministic=True)
                return int(action)
            return fn
        agent_fns[name] = make_fn(model)

    # --- Evaluate WITH early termination ---
    print(f"\n--- EVALUATION: WITH Early Termination ({n_episodes} episodes) ---")
    config_with = EvalConfig(n_episodes=n_episodes, early_termination=True)
    results_with = []
    for name, fn in agent_fns.items():
        print(f"  Evaluating {name}...", end=" ", flush=True)
        t0 = time.time()
        result = evaluate_agent(name, fn, config_with)
        elapsed = time.time() - t0
        print(f"{result['success_rate']:.1f}% success, {result['mean_reward']:.1f} reward ({elapsed:.1f}s)")
        results_with.append(result)

    # --- Evaluate WITHOUT early termination ---
    print(f"\n--- EVALUATION: WITHOUT Early Termination ({n_episodes} episodes) ---")
    config_without = EvalConfig(n_episodes=n_episodes, early_termination=False)
    results_without = []
    for name, fn in agent_fns.items():
        print(f"  Evaluating {name}...", end=" ", flush=True)
        t0 = time.time()
        result = evaluate_agent(name, fn, config_without)
        elapsed = time.time() - t0
        print(f"{result['success_rate']:.1f}% success, {result['mean_reward']:.1f} reward ({elapsed:.1f}s)")
        results_without.append(result)

    # --- Print comparisons ---
    report_with = compare_agents(results_with)
    print(report_with)

    report_without = compare_agents(results_without)
    print(report_without)

    leakage_report = detect_leakage(results_with, results_without)
    print(leakage_report)

    # --- Training time comparison ---
    if train_times:
        print("\n  TRAINING TIME:")
        for algo, t in train_times.items():
            print(f"    {algo}: {t:.0f}s ({t/60:.1f} min)")

    # --- Save results ---
    all_results = {
        "with_early_termination": results_with,
        "without_early_termination": results_without,
        "train_times": train_times,
        "eval_config": {
            "n_episodes": n_episodes,
            "eval_seed_start": EVAL_SEED_START,
            "train_seed_start": TRAIN_SEED_START,
        }
    }
    out_path = RESULTS_DIR / "rigorous_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rigorous RL evaluation")
    parser.add_argument("--quick", action="store_true", help="100 episodes instead of 500")
    parser.add_argument("--train-first", action="store_true", help="Train models before eval")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Training timesteps")
    args = parser.parse_args()

    n_ep = 100 if args.quick else 500
    run_full_eval(n_episodes=n_ep, train_first=args.train_first,
                  train_timesteps=args.timesteps)
