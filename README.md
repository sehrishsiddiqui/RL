# Wildfire Suppression with Reinforcement Learning

**MMAI-845 · Smith School of Business · Queen's University · April 2026 · Team Union**

Can an RL agent — trained from scratch with no prior knowledge — learn to suppress wildfire better than a hand-coded heuristic? Yes, but only with the right algorithm.

## Results

| Agent | Success Rate | Mean Reward | Avg Fire Remaining |
|-------|-------------|-------------|-------------------|
| Random | 10% | 484.3 | 3.80 |
| Greedy (BFS) | 37% | 1269.9 | 2.09 |
| DQN (1M steps) | 35% | 1340.7 | 1.91 |
| **PPO (1M steps)** | **57%** | **1362.3** | **1.20** |

PPO is the only agent that clearly beats the Greedy heuristic. DQN does not — RL must earn its complexity cost.

## Repository Structure

```
├── wildfire_rl/                # Main project code
│   ├── environment/            # ForestFire wrapper with reward shaping
│   ├── agents/                 # Random and Greedy baselines
│   ├── training/               # DQN and PPO training scripts
│   ├── experiments/            # Full pipeline: train + evaluate + sweep
│   ├── utils/                  # Plotting utilities
│   ├── results/                # Saved models, plots, animations
│   └── main.py                # CLI entry point
│
├── gym-cellular-automata/      # ForestFire environment (third-party dependency)
│   └── gym_cellular_automata/
│       └── forest_fire/        # ForestFireHelicopter5x5-v1 environment
│
└── README.md                   # This file
```

- **`wildfire_rl/`** — all of our code: environment wrapper, agents, training, experiments, and results.
- **`gym-cellular-automata/`** — the open-source [gym-cellular-automata](https://github.com/elbecerrasoto/gym-cellular-automata) library by [elbecerrasoto](https://github.com/elbecerrasoto). Provides the base `ForestFireHelicopter5x5-v1` Gymnasium environment that our wrapper builds on. Included here so the project runs without external cloning. Licensed under MIT — see `gym-cellular-automata/LICENSE.txt`.

## Setup

**Prerequisites:** Python 3.11 or 3.12

```bash
# 1. Clone the repository
git clone <repo-url>
cd <repo-name>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -e gym-cellular-automata
pip install -r wildfire_rl/requirements.txt
```

## Quick Start

All commands run from inside the `wildfire_rl/` directory:

```bash
cd wildfire_rl
```

### Run the baselines

```bash
python main.py --agent random --episodes 10
python main.py --agent greedy --episodes 10
```

### Train the RL agents

```bash
python main.py --agent dqn --train      # ~38 min on CPU
python main.py --agent ppo --train      # ~24 min on CPU
```

Both train for 1,000,000 timesteps. Models are saved to `results/`.

### Evaluate a trained model

```bash
python main.py --agent dqn --eval --episodes 100
python main.py --agent ppo --eval --episodes 100
```

### Compare all four agents

```bash
python main.py --compare --episodes 100
```

### Run the full experiment pipeline

Trains both agents, evaluates all four, and runs a learning rate sweep (6 configurations):

```bash
python experiments/run_experiments.py
```

### Generate plots

```bash
python experiments/generate_plots.py
```

Produces 8 visualizations in `results/`: training curves, success rates, mean rewards, position heatmaps, fire-over-time, training time, hyperparameter sweep, and grid snapshots.

## The Environment

- **Grid:** 5×5, cells are empty (0), tree (1), or fire (2)
- **Agent:** Helicopter that extinguishes fire by flying over it
- **Observation:** 34-dimensional vector — grid state, helicopter position, fire distance/direction/density
- **Actions:** 9 discrete — move in 8 directions or stay
- **Reward:** +10 per extinguish, +2× fire reduction, +5 all-clear, −0.1× urgency per step
- **Fire spread:** Stochastic via cellular automata — no fixed pattern to memorize
- **Episode:** 200 steps max. Success = zero fire cells remaining.

## Algorithms

| | DQN | PPO |
|--|-----|-----|
| **Type** | Off-policy, value-based | On-policy, policy gradient |
| **Core idea** | Learn Q(s,a) — value of each action | Learn π(a\|s) — probability of each action |
| **Key mechanism** | Replay buffer (100K transitions) | Clipped surrogate objective (±20%) |
| **Exploration** | Epsilon-greedy: 1.0 → 0.02 | Entropy bonus: 0.01 |
| **Network** | 34 → [256,256] → 9 | Separate actor + critic, [256,256] each |
| **Learning rate** | 1e-4 fixed | 3e-4 with linear decay |
| **Training time** | ~38 min | ~24 min |
| **Result** | 35% success | 57% success |

## Key Findings

1. **PPO wins** — 57% success rate, lowest variance, most trees saved
2. **Greedy is formidable** — 37% with zero training. RL only beats it at PPO, not DQN
3. **Both RL agents learn** — reward grows 2.7× from first to final episodes
4. **DQN ≈ Greedy** — after 1M steps of training, DQN (35%) does not clearly beat a simple rule (37%)

## Built With

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — DQN and PPO implementations
- [Gymnasium](https://gymnasium.farama.org/) — environment API
- [gym-cellular-automata](https://github.com/elbecerrasoto/gym-cellular-automata) — ForestFire base environment
- [Matplotlib](https://matplotlib.org/) — visualization
- [NumPy](https://numpy.org/) — numerical computing

## Team

**Team Union** — MMAI-845 Reinforcement Learning, Queen's University

## License

The `gym-cellular-automata/` directory is third-party code licensed under MIT. See `gym-cellular-automata/LICENSE.txt` for details.
