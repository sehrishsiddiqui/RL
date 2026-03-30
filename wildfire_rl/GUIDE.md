# Wildfire RL — Beginner's Guide

A plain-English walkthrough of every file in this project. No prior RL knowledge assumed.

---

## What is this project?

We train AI agents to fight wildfires on a 5×5 grid. A helicopter flies over burning trees to extinguish them. The goal: put out all fires while preserving as many trees as possible.

We compare **4 different strategies** (called "agents") to see which one is best:

| Agent | Type | How it decides |
|-------|------|----------------|
| **Random** | Baseline | Picks a random direction every step. No intelligence at all. |
| **Greedy** | Heuristic | Finds the nearest fire and beelines toward it. Smart, but not learning. |
| **DQN** | Deep RL | Learns a "value" for each action through trial and error. Uses a neural network. |
| **PPO** | Deep RL | Learns a "policy" (probability of each action) directly. Also uses a neural network. |

---

## The Environment — How the Game Works

**File:** `environment/forest_fire_wrapper.py`

Think of the environment as a board game. The rules are:

### The Grid
```
5×5 grid, each cell is one of:
  0 = empty ground (gray)
  1 = healthy tree (green)
  2 = fire (red)
```

### What the Agent Sees (Observation)
Every step, the agent receives a vector of 30 numbers:
- **25 values** — the 5×5 grid flattened into a list (each cell's state)
- **2 values** — fire probability and tree growth probability (environment physics)
- **2 values** — the helicopter's current row and column
- **1 value** — a "freeze" countdown (explained below)

All values are normalized to [0, 1] so the neural network can process them easily.

### What the Agent Can Do (Actions)
9 possible actions — move in any of 8 directions, or stay put:
```
0: ↖  1: ↑  2: ↗
3: ←  4: ●  5: →
6: ↙  7: ↓  8: ↘
```

**Extinguishing is automatic** — if the helicopter is on a burning cell, the fire goes out. No separate "extinguish" button needed.

### Scoring (Reward)
Every single step, the agent receives:
```
reward = (number of trees − number of fires) / 25
```
- More trees alive = higher reward
- More fires burning = lower reward
- This happens every step, so the agent is constantly motivated to keep trees alive

### The Freeze Mechanic
Fire doesn't spread every single step. There's a countdown:
- When countdown = 0 → fire spreads AND helicopter moves
- When countdown > 0 → only the helicopter moves (fire is "frozen")

This gives the helicopter a few "free" moves between each fire spread — like a head start.

### When Does an Episode End?
Episodes end after **200 steps** (a hard time limit). The environment itself never declares "game over" — we impose this limit ourselves.

**Success** = all fire cells are gone (no cells with value 2 remain).

---

## The Agents — How Each One Decides

### Random Agent
**File:** `agents/random_agent.py`

The simplest possible agent. Every step, it picks a random number from 0–8. That's it.

**Why include it?** It establishes the floor. If a "learning" agent can't beat random, it hasn't learned anything.

```python
def act(self, env) -> int:
    return self.rng.integers(0, self.n_actions)  # random number 0-8
```

### Greedy Baseline
**File:** `agents/greedy_baseline.py`

A hand-coded strategy using **BFS (Breadth-First Search)** — a classic graph algorithm:

1. **Am I on a fire cell?** → Stay (auto-extinguish)
2. **Otherwise:** BFS from my position to find the nearest fire cell
3. **Move one step** along the shortest path toward that fire
4. **Repeat**

This is intelligent but **not learning**. It always does the same thing. It can't adapt or plan ahead — it just chases the nearest fire greedily.

**BFS in plain English:** Imagine dropping a stone in a pond. Ripples spread outward in circles. BFS does the same thing — it checks cells in order of distance from the helicopter until it finds fire.

### DQN (Deep Q-Network)
**File:** `training/train_dqn.py`

DQN learns a **value function** — "how good is it to take action X in state S?"

**How it works:**
1. The agent tries random actions at first (exploration)
2. After each action, it remembers: "I was in state S, took action A, got reward R, ended up in state S'"
3. These memories go into a **replay buffer** (a big list of past experiences)
4. A neural network trains on random samples from this buffer to predict: "given state S, action A has value Q"
5. Over time, the network gets better at predicting which actions lead to high rewards
6. The agent shifts from random exploration to picking the highest-value action (exploitation)

**Key concepts:**
- **Replay buffer** — stores past experiences for re-learning (like studying flashcards)
- **Epsilon-greedy** — starts 100% random, gradually shifts to 95% "best known action"
- **Target network** — a slow-updating copy of the network that stabilizes training
- **Off-policy** — can learn from old experiences (doesn't need fresh data every time)

**Key hyperparameters:**
```python
learning_rate = 1e-4     # how fast the network updates (too high = unstable)
buffer_size = 50_000     # how many experiences to remember
gamma = 0.99             # how much to value future rewards (0.99 = long-term thinker)
exploration_fraction = 0.3  # spend 30% of training exploring randomly
```

### PPO (Proximal Policy Optimization)
**File:** `training/train_ppo.py`

PPO learns a **policy directly** — "in state S, what's the probability of each action?"

**How it works:**
1. The agent collects a batch of experiences by playing 2,048 steps
2. It computes: "how much better/worse was each action than expected?" (advantage)
3. It updates the neural network to make good actions more probable
4. **The key trick:** PPO clips updates so the policy can't change too much at once — this prevents catastrophic "forgetting" where the agent suddenly gets worse

**Key concepts:**
- **On-policy** — only learns from its most recent experiences (no replay buffer)
- **Clipping** — prevents the policy from changing too drastically in one update
- **Entropy bonus** — small reward for being uncertain, which encourages exploration
- **GAE (Generalized Advantage Estimation)** — a technique to reduce noise in advantage estimates

**Key hyperparameters:**
```python
learning_rate = 3e-4     # 3x higher than DQN (on-policy needs faster learning)
n_steps = 2048           # collect this many steps before updating
clip_range = 0.2         # max policy change per update (the "proximal" in PPO)
ent_coef = 0.01          # entropy bonus strength
```

---

## DQN vs PPO — When to Use Which?

| | DQN | PPO |
|--|-----|-----|
| **Learns from** | Past experiences (replay buffer) | Fresh experiences only |
| **Action space** | Best for discrete (like our 9 actions) | Works for both discrete and continuous |
| **Sample efficiency** | Higher (reuses old data) | Lower (needs fresh data) |
| **Stability** | Can be unstable (replay + target net help) | More stable (clipping prevents big swings) |
| **Best for** | Simple environments with clear optimal actions | Complex environments needing exploration |

---

## The Experiment Pipeline

### Training
**File:** `experiments/run_experiments.py`

Runs the complete experiment in 3 steps:

1. **Train** DQN and PPO for 200,000 timesteps each (~15-30 min per agent)
2. **Evaluate** all 4 agents for 100 episodes each, collecting detailed metrics
3. **Hyperparameter sweep** — trains DQN and PPO at 3 different learning rates each to see how sensitive they are

Everything is saved to `results/experiment_results.pkl`.

### Metrics Collected Per Episode
- **Total reward** — sum of all per-step rewards
- **Success** — did all fire get extinguished? (yes/no)
- **Steps** — how many steps the episode lasted
- **Fire remaining** — how many fire cells at the end
- **Trees remaining** — how many tree cells survived
- **Position counts** — how often the helicopter visited each cell (for heatmaps)
- **Fire trace** — fire cells remaining at each timestep (for suppression speed plots)

### Visualizations
**File:** `experiments/generate_plots.py`

Reads the saved results and generates 8 plots:

| # | Plot | What it shows |
|---|------|---------------|
| 01 | Training curves | How DQN and PPO's reward improves during training |
| 02 | Success rate bars | % of episodes where all fire was extinguished |
| 03 | Mean reward bars | Average total reward with error bars |
| 04 | Position heatmaps | Where each agent spends its time on the grid |
| 05 | Fire over time | How fast each agent suppresses fire (fire cells per timestep) |
| 06 | Training time | Wall-clock seconds to train DQN vs PPO |
| 07 | Hyperparameter sweep | How learning rate affects final performance |
| 08 | Grid snapshots | 4-panel visual of an episode at steps 0, 50, 100, 150 |

---

## File Structure

```
wildfire_rl/
│
├── environment/
│   ├── forest_fire_wrapper.py   ← Wraps the pre-built ForestFire environment for our use
│   ├── wildfire_env.py          ← OLD custom environment (not used, kept for reference)
│   └── fire_spread.py           ← OLD fire spread engine (not used, kept for reference)
│
├── agents/
│   ├── random_agent.py          ← Random baseline (picks random actions)
│   └── greedy_baseline.py       ← Greedy BFS baseline (chases nearest fire)
│
├── training/
│   ├── train_dqn.py             ← DQN training script
│   └── train_ppo.py             ← PPO training script
│
├── experiments/
│   ├── run_experiments.py       ← Full experiment pipeline (train + evaluate + sweep)
│   └── generate_plots.py        ← Generate all visualization PNGs
│
├── utils/
│   └── visualization.py         ← Shared plotting utilities
│
├── results/                     ← All outputs: models, rewards, plots
│
├── main.py                      ← CLI entry point
├── requirements.txt             ← Python dependencies
└── GUIDE.md                     ← This file
```

---

## How to Run Everything

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Quick smoke test
```bash
python main.py --agent random --episodes 5
python main.py --agent greedy --episodes 5
```

### Step 3: Train the learning agents
```bash
python main.py --agent dqn --train
python main.py --agent ppo --train
```

### Step 4: Run full experiment (train + evaluate + sweep)
```bash
python experiments/run_experiments.py
```

### Step 5: Generate all plots
```bash
python experiments/generate_plots.py
```

### Step 6: Compare all agents
```bash
python main.py --compare
```

---

## Key RL Concepts Used in This Project

### Markov Decision Process (MDP)
The mathematical framework behind RL. At each step:
- Agent observes **state** (the grid + position)
- Agent takes **action** (move or stay)
- Environment returns **reward** and **next state**
- The goal: maximize total reward over the episode

### Discount Factor (γ = 0.99)
Future rewards are worth slightly less than immediate rewards.
- γ = 1.0 → future rewards worth the same as now
- γ = 0.99 → reward 100 steps away is worth 0.99^100 ≈ 37% of immediate reward
- This prevents the agent from procrastinating

### Exploration vs Exploitation
- **Exploration** — try random actions to discover new strategies
- **Exploitation** — use what you've already learned to maximize reward
- DQN handles this with epsilon-greedy (gradually reduce randomness)
- PPO handles this with entropy bonus (reward for being uncertain)

### Neural Network as Function Approximator
Both DQN and PPO use a **Multi-Layer Perceptron (MLP)** — a simple neural network with:
- Input layer: 30 neurons (our observation vector)
- Hidden layers: 2 layers of 64 neurons each (SB3 default)
- Output layer: 9 neurons (one per action for DQN) or action probabilities (for PPO)

The network learns to map observations to good decisions through thousands of episodes of practice.
