# From Theory to Wildfire: An RL Journey

## How Reinforcement Learning Concepts Led Us to Our Methodology

*Team Union — MMAI 845 Reinforcement Learning*

---

## 1. The Problem: Teaching an Agent to Fight Fire

Imagine you're an incident commander during a wildfire. Every second, you must decide: where do I send the helicopter? The fire is spreading, new fires ignite from lightning, and you can only be in one place at a time. There are no labeled datasets telling you "go left here" — you learn from consequences.

This is exactly the kind of problem Reinforcement Learning was built for. The agent (helicopter) interacts with an environment (burning forest), takes actions (move in 8 directions or stay), observes what happens (fire spreads or gets extinguished), and receives reward signals that tell it how well it's doing.

Our goal: train an RL agent that learns to suppress wildfire on a 5x5 grid — and in the process, deeply understand what makes different RL algorithms tick.

---

## 2. Building Blocks: From Bandits to MDPs

### The Multi-Armed Bandit — Where It All Starts

Before we can fight fires, we need to understand a simpler problem. Imagine standing in front of 9 slot machines (one for each action our helicopter can take). Which one do you pull?

This is the **exploration vs. exploitation dilemma**:
- **Exploit**: Always pull the machine that has paid out the most so far
- **Explore**: Try other machines — maybe one you haven't tried enough is actually better

The **epsilon-greedy** strategy solves this: with probability epsilon, explore randomly; otherwise, exploit the best-known option. This same idea appears in our DQN agent — it starts with high exploration (epsilon = 1.0) and gradually shifts to exploitation (epsilon = 0.02) as it learns which actions work.

**Why this matters for our project:** Our DQN agent uses epsilon-greedy exploration with `exploration_fraction=0.3`, meaning it spends 30% of training exploring before settling into learned behavior. If we explore too little, we might never discover that moving toward fire is good. Too much, and we waste training time on random actions.

### Markov Decision Processes — The Game Board

The bandit problem has no states — you're always standing in front of the same machines. Our wildfire problem has **states that change**: the fire spreads, the helicopter moves, trees regrow.

A **Markov Decision Process (MDP)** formalizes this with four components:
- **States (S)**: Every possible configuration of our 5x5 grid + helicopter position
- **Actions (A)**: 9 discrete movements (8 directions + stay)
- **Transition function p(s'|s,a)**: How the world changes after an action (fire spread, tree regrow)
- **Reward function r(s,a)**: The signal that guides learning

The **Markov property** — the future depends only on the current state, not the history — is why we include the full grid in our observation. The agent doesn't need to remember where the fire was 10 steps ago; it just needs to see the current grid.

**Our observation vector (37 dimensions):**
| Component | Dimensions | What It Captures |
|-----------|-----------|-----------------|
| Grid cells | 25 | Current state of every cell (empty/tree/fire) |
| CA parameters | 2 | Fire spread and tree growth probabilities |
| Helicopter position | 2 | Where the agent is |
| Freeze countdown | 1 | How many free moves before fire spreads again |
| Engineered features | 7 | Fire count, tree count, direction to nearest fire, distance, time progress |

---

## 3. Dynamic Programming: The Ideal We Can't Reach

If we knew exactly how fire spreads — the precise probabilities of every transition — we could solve the MDP perfectly using **Dynamic Programming**. Policy Iteration and Value Iteration guarantee optimal solutions.

But there are two problems:

1. **We don't know the model.** Fire spread in our environment is stochastic. Lightning strikes are random. We can't write down p(s'|s,a) for every possible state.

2. **The state space is enormous.** Even on a 5x5 grid, each cell has 3 possible values (empty/tree/fire), giving 3^25 = 847 billion possible grids — before accounting for helicopter position.

**Issue:** Dynamic Programming needs a known model and visits every state.
**Solution:** Learn from experience instead.

---

## 4. Monte Carlo Methods: Learning from Complete Episodes

The first model-free approach: **Monte Carlo (MC) prediction**. The idea is simple — play complete episodes, observe the total return from each state-action pair, and average them.

For our wildfire environment, one MC episode would be:
1. Start with some fire configuration
2. Take actions for 200 steps
3. After the episode ends, look back: "From state S at step 50, my total return was G. From state S at step 100, my total return was G'."
4. Average these returns across many episodes

**The MC update rule:**
```
Q(S_t, A_t) ← Q(S_t, A_t) + α * (G_t - Q(S_t, A_t))
```

**Issue with MC for our problem:**
- We must wait until the episode ends (200 steps) before learning anything
- In our environment, fire dynamics are highly stochastic — returns have enormous variance
- Early episodes provide almost no useful signal because the agent acts randomly for so long

**We need something that learns from every single step, not just at the end.**

---

## 5. Temporal Difference Learning: Learn As You Go

**TD Learning** is the key innovation that makes practical RL possible. Instead of waiting for the actual return G_t, we **bootstrap** — use our current estimate of the next state's value as a stand-in:

```
Q(S_t, A_t) ← Q(S_t, A_t) + α * [R_{t+1} + γ * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
```

The term `R_{t+1} + γ * Q(S_{t+1}, A_{t+1})` is the **TD target** — one step of real experience plus an estimate of what comes next.

**Why this matters for wildfire suppression:** The agent doesn't need to play out a full 200-step episode to learn that "moving toward fire is good." After just one step where it moves closer to fire and gets a proximity reward, it immediately updates its value estimates.

### SARSA: On-Policy TD Control

**SARSA** (State-Action-Reward-State-Action) is an on-policy TD algorithm. It learns the value of the policy it's actually following — including its exploratory actions.

```
Choose A_t using epsilon-greedy on Q
Take A_t, observe R_{t+1}, S_{t+1}
Choose A_{t+1} using epsilon-greedy on Q
Q(S_t, A_t) ← Q(S_t, A_t) + α * [R_{t+1} + γ * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
```

Because SARSA evaluates the policy it follows (including random exploration), it learns "safe" policies that account for the possibility of making mistakes. In a wildfire context, SARSA might learn to stay near fire clusters rather than making bold moves — because it factors in the chance of randomly moving away.

### Q-Learning: Off-Policy — Learn the Best, Do Something Else

**Q-Learning** decouples what the agent does (behavior policy) from what it learns about (target policy). It always updates toward the **best possible** next action, regardless of what it actually does:

```
Q(S_t, A_t) ← Q(S_t, A_t) + α * [R_{t+1} + γ * max_a' Q(S_{t+1}, a') - Q(S_t, A_t)]
```

The `max` is the key difference from SARSA. Q-Learning asks "what's the best thing I could do from here?" rather than "what will I probably do from here?"

**This is why we chose Q-Learning (DQN) as one of our algorithms.** In a wildfire environment where the agent needs to explore different strategies, Q-Learning can learn the optimal fire-fighting policy while still exploring. SARSA would learn a more conservative policy influenced by its own exploration noise.

---

## 6. The Scaling Problem: Why Tabular Methods Break

All the algorithms above store Q-values in a **table** — one entry per (state, action) pair. On our 5x5 grid with 9 actions, the theoretical table size is 3^25 × 9 = ~7.6 trillion entries.

Even practically, the agent must visit each state-action pair many times for convergence. With stochastic fire dynamics, many states are visited rarely or never.

**Issue:** Tabular methods don't scale and can't generalize.
**Solution:** Function approximation — use a function (not a table) to estimate Q-values.

### From Tables to Functions

Instead of storing Q(s,a) for every possible (s,a), we learn a **parameterized function** Q(s, a, w) where w are learnable weights. States with similar features get similar Q-values — learning about one fire configuration helps with similar ones.

**Linear approximation:**
```
Q(s, a, w) = w_1 * x_1(s) + w_2 * x_2(s) + ... + w_k * y(a)
```

**The gradient descent update (approximate Q-learning):**
```
w ← w + α * [R_{t+1} + γ * max_a' Q(S_{t+1}, a', w) - Q(S_t, A_t, w)] * ∇Q(S_t, A_t, w)
```

This is exactly the tabular update, but instead of modifying one table entry, we adjust all weights — which affects Q-values for similar states too. This is **generalization**: learning about fighting a fire in the top-left corner helps with fighting a fire in the top-right corner.

---

## 7. Deep Q-Networks (DQN): Our First Algorithm

Linear functions can only capture simple patterns. **Neural networks** can approximate arbitrarily complex functions — and this is where Deep RL begins.

**DQN** (Mnih et al., 2015) combines Q-Learning with a deep neural network:
- Input: State features (our 37-dim observation vector)
- Network: Two hidden layers of 256 neurons each (ReLU activation)
- Output: 9 Q-values, one per action
- Action selection: epsilon-greedy on the Q-values

But naively plugging a neural network into Q-Learning breaks. Two critical problems emerge:

### Problem 1: Correlated Data (Not IID)

In supervised learning, training data is Independent and Identically Distributed (IID) — a cat image is equally likely at any training step. In RL, consecutive experiences are **highly correlated**: step 50 looks a lot like step 51. This correlation destabilizes neural network training.

**Solution: Experience Replay**
Store transitions (s, a, r, s') in a large buffer (200,000 in our setup). When training, sample random mini-batches from this buffer. This breaks temporal correlation and makes the data more IID.

### Problem 2: Moving Target

When we update our network weights w, it changes both our current prediction Q(s,a,w) AND the target R + γ * max Q(s',a',w). We're trying to hit a target that moves every time we adjust our aim.

**Solution: Target Network**
Maintain a separate "target network" with frozen weights w'. Update targets using w', and only copy w → w' every N steps (500 in our setup). This stabilizes learning by keeping the target fixed for extended periods.

### Our DQN Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| Network | [256, 256] | Sufficient capacity for 37-dim input |
| Learning rate | 1e-4 (decaying) | Stable learning; decays to avoid oscillation |
| Buffer size | 200,000 | Large replay buffer for diverse experience |
| Batch size | 256 | Smooth gradient estimates |
| Gamma | 0.99 | Values immediate fire suppression highly |
| Exploration | 30% fraction, final epsilon 0.02 | Enough exploration, then mostly exploit |
| Target update | Every 500 steps | Balance between stability and learning speed |

---

## 8. PPO: Our Second Algorithm — A Different Philosophy

DQN learns a **value function** (Q-values) and derives a policy from it. **Proximal Policy Optimization (PPO)** takes a fundamentally different approach: it directly optimizes the **policy** — the probability distribution over actions.

### Why Policy Gradient Methods?

Value-based methods like DQN have a limitation: they must take the argmax over actions to derive a policy. This works for discrete actions but struggles with:
- Continuous action spaces
- Stochastic policies (where randomness is desirable)
- Problems where small changes in Q-values cause large policy shifts

Policy gradient methods directly parameterize the policy π(a|s, θ) and optimize it using gradient ascent on expected returns.

### The PPO Innovation: Controlled Updates

The predecessor, TRPO (Trust Region Policy Optimization), constrains how much the policy can change per update. PPO simplifies this with a clipping mechanism:

```
L(θ) = min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)
```

Where r_t(θ) is the ratio of new policy to old policy probability. The clipping (ε = 0.2 in our setup) prevents the policy from changing too drastically in one update — it can improve, but not by more than 20% in either direction.

### On-Policy vs. Off-Policy: A Key Distinction

| Aspect | DQN (Off-Policy) | PPO (On-Policy) |
|--------|-------------------|-----------------|
| Data usage | Replays old experience from buffer | Uses only fresh experience, then discards |
| Sample efficiency | More efficient (reuses data) | Less efficient (data used once) |
| Stability | Can diverge with function approx. | More stable due to trust region |
| Exploration | Epsilon-greedy (add randomness to action selection) | Entropy bonus (add randomness to policy itself) |

### Why Both Algorithms for Wildfire?

Comparing DQN and PPO reveals fundamental trade-offs:

1. **DQN's replay buffer** is ideal for our stochastic environment — rare events (successful fire suppression) stay in the buffer and get replayed many times. PPO discards this valuable experience after one use.

2. **PPO's stability** matters because our reward function is complex (multiple components). DQN can oscillate when reward signals conflict; PPO's clipped updates prevent catastrophic policy changes.

3. **The fire environment is non-stationary** — the distribution of states changes as the agent improves. DQN's replay buffer contains stale data from the old policy. PPO always trains on fresh data matching the current policy.

### Our PPO Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| Network | Separate pi=[256,256], vf=[256,256] | Decoupled policy and value heads |
| Learning rate | 3e-4 (decaying) | Higher than DQN; PPO is more stable |
| Rollout length | 2,048 steps | Long enough for meaningful advantage estimates |
| Batch size | 128 | Balance between noise and computation |
| Epochs per rollout | 10 | Reuse each rollout moderately |
| Gamma | 0.99 | Same as DQN for fair comparison |
| GAE lambda | 0.95 | Generalized Advantage Estimation bias-variance |
| Entropy coefficient | 0.05 | Encourage exploration through policy randomness |
| Clip range | 0.2 | Prevent destructive policy updates |
| Parallel envs | 4 | Diverse experience for on-policy learning |

---

## 9. The Reward Function: The Heart of the Problem

The most impactful design decision in any RL project is the **reward function**. It's the only way we communicate our goals to the agent. Get it wrong, and the agent optimizes for something we didn't intend.

### Version 1 (Original): Too Weak

```
reward = (trees - fires) / 25
if hit: reward += 2.0
if fire_exists: reward += 0.5 / (1 + distance_to_fire)
```

**Problems:**
- Base reward is dominated by stochastic fire spawning noise
- No terminal signal for success or failure
- Agent couldn't distinguish "almost cleared all fires" from "fires everywhere"

**Result:** DQN 13% success, PPO 19% success. Barely above Random (11%).

### Version 2 (Redesigned): Clear Signals

```
reward = 0.0  (start clean)
if all_fires_out: reward += 10.0 + early_finish_bonus  (massive success signal)
if hit: reward += 3.0                                   (direct suppression reward)
reward += (prev_fires - current_fires) * 1.0            (fire delta — reduction is good)
if fire_exists: reward += tiered_proximity               (0.5/0.3/0.1 by distance)
reward -= 0.05                                          (step penalty — don't dawdle)
if episode_ends_with_fire: reward -= 2.0 * fire_count   (failure penalty)
```

**Key design principles:**
1. **Sparse vs. dense rewards:** The terminal bonus (sparse) gives direction; the per-step signals (dense) give gradient. Both are needed.
2. **Reward shaping must not change the optimal policy:** Our fire-delta reward is a **potential-based shaping** — it rewards progress without creating local optima.
3. **Magnitude matters:** The +10 success bonus must dominate the cumulative step penalties, otherwise the agent learns to farm proximity rewards instead of actually clearing fires.

---

## 10. What We Learned: Evaluation Integrity

### The Data Leakage Problem

In supervised learning, you never test on training data. In RL, "data" is episodes generated from seeds. Our original evaluation used seeds 42-141, but training used seeds 42-45. The first four test episodes were identical to training — the agent had memorized them.

**Fix:** Strict seed protocol:
- Training: seeds 0-999
- Validation (during training): seeds 1000-1999
- Testing (final evaluation): seeds 10000+ (zero overlap)

### The Early Termination Question

We added early termination: if all fires are cleared, the episode ends. This dramatically boosts success rates (even Random achieves 84-94%) because the agent only needs to clear the initial fires before lightning strikes again.

**Without early termination** (the honest test), the agent must maintain a fire-free grid for the full 200 steps despite random lightning strikes — a much harder problem and the fair comparison to our original results.

### The Observation Engineering Trade-off

We added features like "direction to nearest fire" and "distance to fire." This is the same information the Greedy baseline uses (BFS to nearest fire). It helps the RL agents learn faster, but it also means they're partially being hand-fed the solution rather than discovering it from raw grid observations.

For a course project, this is acceptable and demonstrates understanding of **feature engineering for RL**. For a research paper, you'd want to show the agent can learn from raw observations alone.

---

## 11. Algorithm Comparison: What We Expected vs. What Happened

### Theoretical Predictions

| Factor | Favors DQN | Favors PPO |
|--------|-----------|-----------|
| Stochastic environment | Replay buffer stores rare successes | Fresh data matches current dynamics |
| Small discrete action space (9) | Efficient Q-value estimation | No advantage over value-based |
| Short episodes (100-200 steps) | Sufficient replay diversity | Rollouts capture full episodes |
| Complex reward function | Off-policy handles multi-component rewards | Clipped updates prevent oscillation |

### What Our Results Show

*(Results from rigorous evaluation with disjoint seeds, no early termination)*

**DQN outperforms PPO** in sustained fire suppression. This aligns with theory:
1. The **replay buffer** is crucial — successful fire-clearing episodes are rare and get replayed many times, giving DQN more learning signal per episode
2. The **small action space** (9 actions) means DQN's max-over-actions is computationally cheap and accurate
3. PPO's on-policy nature means it discards hard-won experience after one use — wasteful in a stochastic environment where success is rare

**Greedy beats both RL agents** because:
1. It has direct grid access (not an observation vector) and uses BFS (optimal pathfinding)
2. It reacts optimally every single step with zero learning delay
3. On a 5x5 grid, there isn't enough complexity for learned strategies to outperform hand-coded heuristics
4. The advantage of RL would emerge on larger grids where greedy heuristics break down (e.g., choosing which fire to prioritize when multiple are burning)

---

## 12. The Concept Chain: How Everything Connects

```
Multi-Armed Bandit (exploration vs exploitation)
    │
    ├─ Issue: Only one state, need multiple states
    ▼
Markov Decision Process (formalize states, actions, transitions)
    │
    ├─ Issue: Need a known model to solve with DP
    ▼
Monte Carlo Methods (learn from complete episodes without a model)
    │
    ├─ Issue: Must wait for episode end; high variance; no continuing tasks
    ▼
Temporal Difference Learning (learn every step via bootstrapping)
    │
    ├─ SARSA (on-policy) ──── learns the policy you follow
    ├─ Q-Learning (off-policy) ──── learns the optimal policy
    │
    ├─ Issue: Tabular methods don't scale to large state spaces
    ▼
Function Approximation (generalize across similar states)
    │
    ├─ Linear (simple, convergence guaranteed)
    ├─ Neural Networks (powerful, but unstable)
    │
    ├─ Issue: Correlated data + moving targets break neural nets
    ▼
DQN (experience replay + target networks stabilize deep Q-learning)
    │
    ├─ Issue: Value-based methods have limitations (argmax, discrete only)
    ▼
PPO (directly optimize policy with clipped updates for stability)
```

Each step in this chain solves a specific limitation of the previous approach. Our project implements the final two nodes (DQN and PPO) and validates them against simpler baselines (Random, Greedy) to demonstrate that the added complexity provides measurable benefit — or, in some cases, that it doesn't, which is equally informative.

---

## 13. Conclusion: What This Project Demonstrates

1. **RL is not magic.** A hand-coded greedy heuristic beats sophisticated deep RL on a small grid. RL's value proposition is scalability and adaptability — advantages that emerge on larger, more complex problems.

2. **The reward function is everything.** Our initial reward (13% success) vs. redesigned reward (22%+ success) shows that RL algorithms are only as good as the signal they optimize.

3. **Evaluation integrity matters.** Seed overlap, early termination, and observation engineering can all inflate results. Rigorous evaluation with disjoint seeds and controlled conditions is essential.

4. **Off-policy (DQN) vs. on-policy (PPO) is a real trade-off.** In our stochastic environment with rare successes, DQN's replay buffer provides a meaningful advantage. PPO's stability helps during training, but its inability to reuse experience hurts.

5. **The course concepts form a coherent chain.** Every concept from bandits to DQN solves a specific problem introduced by the previous one. Our project is a concrete instantiation of this entire chain, applied to a problem with real-world motivation.

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
- Givigi, S. (2026). MMAI 845 Reinforcement Learning Lectures 1-5. Queen's University.
