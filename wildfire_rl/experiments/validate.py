"""
validate.py
Self-evaluation framework — validates the entire codebase, training results,
and outputs. Run this to check everything is working before final submission.

Usage:
    python experiments/validate.py           # full validation
    python experiments/validate.py --quick   # smoke test only (no training check)
"""

import sys
import argparse
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"

PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"
WARN = "\033[93m WARN \033[0m"
SKIP = "\033[94m SKIP \033[0m"

results_log = []


def check(name: str, fn, critical: bool = True):
    """Run a validation check and log result."""
    try:
        passed, detail = fn()
        if passed:
            print(f"  [{PASS}] {name}")
            results_log.append(("PASS", name, detail))
        else:
            tag = FAIL if critical else WARN
            print(f"  [{tag}] {name} — {detail}")
            results_log.append(("FAIL" if critical else "WARN", name, detail))
    except Exception as e:
        tag = FAIL if critical else WARN
        print(f"  [{tag}] {name} — Exception: {e}")
        results_log.append(("FAIL" if critical else "WARN", name, str(e)))


# =====================================================================
# Phase 1: Import & Environment Checks
# =====================================================================

def check_imports():
    def _check():
        import gymnasium
        import stable_baselines3
        import gym_cellular_automata
        import matplotlib
        return True, "All imports OK"
    return _check


def check_env_creation():
    def _check():
        from environment.forest_fire_wrapper import ForestFireWrapper
        env = ForestFireWrapper(max_steps=200, seed=42)
        obs, info = env.reset(seed=42)
        assert obs.shape == (30,), f"Expected obs shape (30,), got {obs.shape}"
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
        assert env.action_space.n == 9, f"Expected 9 actions, got {env.action_space.n}"
        env.close()
        return True, f"Obs shape={obs.shape}, actions={env.action_space.n}"
    return _check


def check_env_episode():
    def _check():
        from environment.forest_fire_wrapper import ForestFireWrapper
        env = ForestFireWrapper(max_steps=200, seed=42)
        obs, _ = env.reset(seed=42)
        total_reward = 0
        for _ in range(200):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
            if terminated or truncated:
                break
        env.close()
        assert "fire_remaining" in info, "Missing fire_remaining in info"
        assert "success" in info, "Missing success in info"
        assert "step" in info, "Missing step in info"
        return True, f"Episode ran {info['step']} steps, reward={total_reward:.2f}"
    return _check


def check_sb3_compatibility():
    def _check():
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
        from environment.forest_fire_wrapper import ForestFireWrapper
        env = Monitor(ForestFireWrapper(max_steps=200, seed=42))
        check_env(env, warn=False)
        env.close()
        return True, "SB3 env check passed"
    return _check


# =====================================================================
# Phase 2: Agent Checks
# =====================================================================

def check_random_agent():
    def _check():
        from agents.random_agent import RandomAgent
        from environment.forest_fire_wrapper import ForestFireWrapper
        env = ForestFireWrapper(max_steps=200, seed=42)
        agent = RandomAgent(n_actions=9, seed=42)
        obs, _ = env.reset(seed=42)
        actions = [agent.act(env) for _ in range(100)]
        env.close()
        assert all(0 <= a < 9 for a in actions), "Actions out of range"
        assert len(set(actions)) > 1, "All actions identical — not random"
        return True, f"100 actions, {len(set(actions))} unique values"
    return _check


def check_greedy_agent():
    def _check():
        from agents.greedy_baseline import GreedyBaselineAgent
        from environment.forest_fire_wrapper import ForestFireWrapper
        env = ForestFireWrapper(max_steps=200, seed=42)
        agent = GreedyBaselineAgent()
        obs, _ = env.reset(seed=42)
        total_reward = 0
        done = False
        while not done:
            action = agent.act(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        env.close()
        return True, f"Reward={total_reward:.2f}, success={info.get('success')}"
    return _check


def check_greedy_beats_random():
    def _check():
        from agents.random_agent import RandomAgent
        from agents.greedy_baseline import GreedyBaselineAgent
        from environment.forest_fire_wrapper import ForestFireWrapper

        random_rewards, greedy_rewards = [], []
        for ep in range(20):
            # Random
            env = ForestFireWrapper(max_steps=200, seed=42)
            agent = RandomAgent(n_actions=9, seed=ep)
            obs, _ = env.reset(seed=ep)
            r = 0
            done = False
            while not done:
                obs, reward, term, trunc, _ = env.step(agent.act(env))
                r += reward
                done = term or trunc
            random_rewards.append(r)
            env.close()

            # Greedy
            env = ForestFireWrapper(max_steps=200, seed=42)
            agent = GreedyBaselineAgent()
            obs, _ = env.reset(seed=ep)
            r = 0
            done = False
            while not done:
                obs, reward, term, trunc, _ = env.step(agent.act(env))
                r += reward
                done = term or trunc
            greedy_rewards.append(r)
            env.close()

        rm = np.mean(random_rewards)
        gm = np.mean(greedy_rewards)
        passed = gm > rm
        return passed, f"Greedy={gm:.2f} vs Random={rm:.2f} ({'Greedy wins' if passed else 'Random wins — problem!'})"
    return _check


# =====================================================================
# Phase 3: Training Output Checks
# =====================================================================

def check_trained_model(algo: str):
    def _check():
        model_path = RESULTS_DIR / f"{algo.lower()}_best" / "best_model.zip"
        if not model_path.exists():
            return False, f"Model not found at {model_path}"
        size_mb = model_path.stat().st_size / (1024 * 1024)
        return True, f"Model exists ({size_mb:.2f} MB)"
    return _check


def check_training_rewards(algo: str):
    def _check():
        rpath = RESULTS_DIR / f"{algo.lower()}_episode_rewards.npy"
        if not rpath.exists():
            return False, f"Rewards file not found: {rpath}"
        rewards = np.load(str(rpath))
        if len(rewards) < 50:
            return False, f"Only {len(rewards)} episodes — too few"
        early = np.mean(rewards[:50])
        late = np.mean(rewards[-50:])
        improved = late > early
        return improved, (f"{len(rewards)} episodes, early={early:.2f}, late={late:.2f} "
                         f"({'improved' if improved else 'NO improvement — check hyperparams'})")
    return _check


def check_model_eval(algo: str):
    def _check():
        from stable_baselines3 import DQN, PPO
        from environment.forest_fire_wrapper import ForestFireWrapper

        model_path = RESULTS_DIR / f"{algo.lower()}_best" / "best_model"
        if not model_path.with_suffix(".zip").exists():
            return False, "Model not found"

        ModelClass = DQN if algo == "DQN" else PPO
        model = ModelClass.load(str(model_path))
        env = ForestFireWrapper(max_steps=200, seed=99)

        rewards = []
        for ep in range(10):
            obs, _ = env.reset(seed=99 + ep)
            r = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = env.step(action)
                r += reward
                done = term or trunc
            rewards.append(r)

        env.close()
        mean_r = np.mean(rewards)
        return True, f"10-episode eval: mean_reward={mean_r:.2f}"
    return _check


# =====================================================================
# Phase 4: Results & Visualization Checks
# =====================================================================

def check_experiment_results():
    def _check():
        import pickle
        path = RESULTS_DIR / "experiment_results.pkl"
        if not path.exists():
            return False, "experiment_results.pkl not found — run experiments first"
        with open(path, "rb") as f:
            data = pickle.load(f)
        agents_found = [k for k in ["Random", "Greedy", "DQN", "PPO"] if k in data]
        has_sweep = "sweep" in data
        has_training = "training" in data
        return True, f"Agents: {agents_found}, sweep={has_sweep}, training={has_training}"
    return _check


def check_plots_generated():
    def _check():
        expected = [
            "01_training_curves.png",
            "02_success_rate.png",
            "03_mean_reward.png",
            "04_position_heatmaps.png",
            "05_fire_over_time.png",
            "06_training_time.png",
            "07_hyperparam_sweep.png",
            "08_grid_snapshots.png",
        ]
        found = []
        missing = []
        for f in expected:
            if (RESULTS_DIR / f).exists():
                found.append(f)
            else:
                missing.append(f)
        passed = len(missing) == 0
        detail = f"{len(found)}/{len(expected)} plots found"
        if missing:
            detail += f". Missing: {', '.join(missing)}"
        return passed, detail
    return _check


# =====================================================================
# Phase 5: Quality Checks
# =====================================================================

def check_learning_agents_beat_random():
    def _check():
        import pickle
        path = RESULTS_DIR / "experiment_results.pkl"
        if not path.exists():
            return False, "No experiment results"
        with open(path, "rb") as f:
            data = pickle.load(f)
        if "Random" not in data:
            return False, "Random results missing"
        random_reward = data["Random"]["mean_reward"]
        issues = []
        for algo in ["DQN", "PPO"]:
            if algo in data:
                algo_reward = data[algo]["mean_reward"]
                if algo_reward <= random_reward:
                    issues.append(f"{algo} ({algo_reward:.2f}) <= Random ({random_reward:.2f})")
        if issues:
            return False, "; ".join(issues) + " — agents not learning!"
        return True, "Both DQN and PPO beat Random"
    return _check


def check_learning_agents_beat_greedy():
    def _check():
        import pickle
        path = RESULTS_DIR / "experiment_results.pkl"
        if not path.exists():
            return False, "No experiment results"
        with open(path, "rb") as f:
            data = pickle.load(f)
        if "Greedy" not in data:
            return False, "Greedy results missing"
        greedy_reward = data["Greedy"]["mean_reward"]
        status = []
        for algo in ["DQN", "PPO"]:
            if algo in data:
                r = data[algo]["mean_reward"]
                beats = r > greedy_reward
                status.append(f"{algo}={r:.2f} {'>' if beats else '<='} Greedy={greedy_reward:.2f}")
        return True, "; ".join(status) + " (beating Greedy is aspirational, not required)"
    return _check


# =====================================================================
# Runner
# =====================================================================

def run_validation(quick: bool = False):
    print("\n" + "=" * 60)
    print("  WILDFIRE RL — PROJECT VALIDATION")
    print("=" * 60)

    print("\n--- Phase 1: Environment ---")
    check("Python imports", check_imports(), critical=True)
    check("Environment creation", check_env_creation(), critical=True)
    check("Full episode run", check_env_episode(), critical=True)
    check("SB3 compatibility", check_sb3_compatibility(), critical=True)

    print("\n--- Phase 2: Agents ---")
    check("Random agent", check_random_agent(), critical=True)
    check("Greedy agent", check_greedy_agent(), critical=True)
    check("Greedy > Random (20 episodes)", check_greedy_beats_random(), critical=True)

    if quick:
        print(f"\n  [{SKIP}] Skipping training/results checks (--quick mode)")
    else:
        print("\n--- Phase 3: Training Outputs ---")
        check("DQN model saved", check_trained_model("DQN"), critical=True)
        check("PPO model saved", check_trained_model("PPO"), critical=True)
        check("DQN learning curve", check_training_rewards("DQN"), critical=True)
        check("PPO learning curve", check_training_rewards("PPO"), critical=True)
        check("DQN eval runs", check_model_eval("DQN"), critical=True)
        check("PPO eval runs", check_model_eval("PPO"), critical=True)

        print("\n--- Phase 4: Results & Plots ---")
        check("Experiment results file", check_experiment_results(), critical=True)
        check("All 8 plots generated", check_plots_generated(), critical=True)

        print("\n--- Phase 5: Quality ---")
        check("DQN & PPO beat Random", check_learning_agents_beat_random(), critical=True)
        check("DQN & PPO vs Greedy", check_learning_agents_beat_greedy(), critical=False)

    # Summary
    passes = sum(1 for s, _, _ in results_log if s == "PASS")
    fails  = sum(1 for s, _, _ in results_log if s == "FAIL")
    warns  = sum(1 for s, _, _ in results_log if s == "WARN")
    total  = len(results_log)

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passes}/{total} passed, {fails} failed, {warns} warnings")
    print("=" * 60)

    if fails > 0:
        print("\n  Failed checks:")
        for status, name, detail in results_log:
            if status == "FAIL":
                print(f"    - {name}: {detail}")
        print("\n  Fix the above issues before submission.")
    else:
        print("\n  All critical checks passed. Ready for submission.")

    return fails == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Wildfire RL project")
    parser.add_argument("--quick", action="store_true", help="Smoke test only")
    args = parser.parse_args()
    success = run_validation(quick=args.quick)
    sys.exit(0 if success else 1)
