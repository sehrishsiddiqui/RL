# MMAI-845 — Wildfire Containment RL Project
## Project Plan & Progress Tracker

**Project:** Autonomous Firefighting Resource Routing via Reinforcement Learning
**Proposal Due:** March 24, 2026
**Project Window:** March 24 – April 20, 2026
**Team:** XXXXX

---

## Project Summary

We use a modified Maze environment with a custom fire spread cellular automaton to train an RL agent to route firefighting resources and establish containment lines before a spreading wildfire reaches populated zones. We compare DQN vs PPO against a greedy shortest-path baseline.

**Research Question:**
> Does the RL agent learn a proactive containment strategy (intercept fire front early) or a reactive one (chase the fire), and how does wind direction affect the learned policy?

---

## Codebase Structure

```
wildfire_rl/
  environment/
    wildfire_env.py       # Custom Gym environment (main env wrapper)
    fire_spread.py        # Cellular automaton fire spread module
  agents/
    greedy_baseline.py    # BFS-based greedy baseline agent
  utils/
    visualization.py      # Heatmaps, rendering, episode replay
  training/
    train_dqn.py          # DQN training script
    train_ppo.py          # PPO training script
  results/                # Saved models, logs, plots
  main.py                 # Entry point — run any agent
  requirements.txt
```

---

## Week 1 — Environment Setup & Baseline
**Dates:** March 24–30
**Goal:** Working custom Gym environment + baseline agent before any RL code

### Tasks
- [ ] Set up repo and virtual environment
- [ ] Install dependencies (`gym`, `stable-baselines3`, `numpy`, `matplotlib`, `pygame`)
- [ ] Build fire spread module (`fire_spread.py`)
  - [ ] Cellular automaton: fire spreads to adjacent cells each timestep
  - [ ] Wind direction vector biases spread probability
  - [ ] Burned cells become permanently impassable walls
  - [ ] Populated zone cells defined as fixed high-value targets
- [ ] Build custom Gym environment (`wildfire_env.py`)
  - [ ] `reset()` — initialize grid, fire start, agent position
  - [ ] `step()` — apply action, spread fire, compute reward
  - [ ] `render()` — visual output for debugging
  - [ ] State: agent position + fire map + wind vector + resources remaining
  - [ ] Actions: Move N/S/E/W (0–3) + Deploy resource (4) + Hold (5)
  - [ ] Reward function implemented correctly
- [ ] Implement greedy shortest-path baseline (`greedy_baseline.py`)
  - [ ] BFS to nearest fire-adjacent unburned cell
  - [ ] Deploy when adjacent to fire front
- [ ] Run baseline agent, record initial metrics
- [ ] Verify Gym environment passes SB3 compatibility check

### Reward Function
| Event | Reward |
|---|---|
| Each timestep | −1 |
| Containment line placed successfully | +50 |
| Fire reaches populated zone | −100 |
| Full containment achieved | +200 |
| Resource deployed on already-burned cell | −20 |

### Deliverable
Working `wildfire_env.py` + `fire_spread.py` + baseline agent with logged results

### Notes / Blockers
>

---

## Week 2 — Train DQN and PPO
**Dates:** March 31 – April 6
**Goal:** Both algorithms training stably, initial comparison logged

### Tasks
- [ ] Train DQN (Stable-Baselines3)
  - [ ] Start on small 8×8 grid — verify learning signal
  - [ ] Scale to full grid once stable
  - [ ] Tune: learning rate, buffer size, exploration schedule
  - [ ] Log: episode reward, containment rate, fire reach rate
- [ ] Train PPO (Stable-Baselines3)
  - [ ] Same grid setup as DQN for fair comparison
  - [ ] Tune: learning rate, entropy coefficient, clip range
  - [ ] Log same metrics as DQN
- [ ] Set up TensorBoard or matplotlib training curves
- [ ] Reward shaping sanity checks
  - [ ] Verify agent isn't spamming deploy on burned cells
  - [ ] Verify −1/step is creating urgency (agent moves, doesn't hold indefinitely)
- [ ] Run initial 3-way comparison: DQN vs PPO vs Greedy Baseline
- [ ] Document hyperparameters used for reproducibility

### Key Hyperparameters to Track
| Parameter | DQN | PPO |
|---|---|---|
| Learning rate | | |
| Batch size | | |
| γ (gamma) | | |
| Training steps | | |
| Other | | |

### Deliverable
Training curves for both algorithms + initial comparison table (mean reward, containment rate, timesteps/episode)

### Notes / Blockers
>

---

## Week 3 — Experiments & Analysis
**Dates:** April 7–13
**Goal:** Answer the research question with experimental evidence

### Experiment Matrix
| # | Variable | Values | Question |
|---|---|---|---|
| 1 | Wind direction | N / S / E / W | Does agent generalize or overfit to one direction? |
| 2 | Grid size | 8×8 / 16×16 | Does policy degrade on larger terrain? |
| 3 | Fire spread rate | Slow / Medium / Fast | At what speed does RL fail vs greedy? |
| 4 | Resource budget | Scarce / Abundant | Does agent learn to conserve resources? |
| 5 | Populated zone placement | Edge / Center / Corner | Does agent learn to prioritize high-value zones? |

### Analysis Tasks
- [ ] Generate agent position heatmaps — proactive vs reactive classification
- [ ] Compare DQN vs PPO across all 5 experiment dimensions
- [ ] Identify where greedy baseline beats RL (document honestly)
- [ ] Plot containment success rate vs fire spread speed for all three agents
- [ ] Wind direction generalization test — train on one direction, test on all four

### Key Metrics to Report
| Metric | Greedy | DQN | PPO |
|---|---|---|---|
| Containment success rate | | | |
| Mean episode reward | | | |
| Avg timesteps to containment | | | |
| Fire reach rate (populated zones) | | | |
| Resources wasted (%) | | | |

### Deliverable
Full results table + heatmap visualizations + written analysis

### Notes / Blockers
>

---

## Week 4 — Report & Presentation
**Dates:** April 14–20
**Goal:** Final submission — report, slides, clean codebase

### Report Tasks
- [ ] Problem Description section (30% of grade)
  - [ ] Environment description with fire spread modification
  - [ ] State space, action space, reward function
  - [ ] Algorithm selection rationale (DQN vs PPO)
- [ ] Business Plan section (70% of grade)
  - [ ] BC Wildfire Service / CAL FIRE / USDA Forest Service case
  - [ ] Catastrophe insurance angle (Swiss Re, Munich Re, FM Global)
  - [ ] Autonomous aerial firefighting (Joby Aviation, DARPA programs)
- [ ] Results section
  - [ ] Learning curves
  - [ ] Comparison table
  - [ ] Heatmap visualizations
- [ ] Discussion: proactive vs reactive strategy finding, limitations, future work
- [ ] Format to under 1 page (final PDF/DOCX)

### Presentation Tasks
- [ ] Slide 1: Problem — wildfire routing, BC 2023 stats ($720M, 2.84M hectares)
- [ ] Slide 2: Environment demo / fire spread animation (best visual asset)
- [ ] Slide 3: Methodology — state/action/reward, DQN vs PPO
- [ ] Slide 4: Results — comparison table + learning curves
- [ ] Slide 5: Business case — three commercial applications
- [ ] Slide 6: Conclusion + limitations

### Code Cleanup Tasks
- [ ] Comment fire spread module clearly
- [ ] README with setup and run instructions
- [ ] Single entry point: `python main.py --agent [dqn/ppo/greedy]`
- [ ] All results reproducible with fixed random seed

### Deliverable
Final report (PDF/DOCX) + presentation deck + clean codebase

### Notes / Blockers
>

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Fire spread module unstable (fire dies or explodes) | Medium | High | Test cellular automaton independently first |
| DQN/PPO flat reward curve (not learning) | Medium | High | Start 4×4 grid, verify signal before scaling |
| Reward hacking (agent finds exploit) | Low | Medium | Add render early — watch what agent actually does |
| SB3 compatibility issue with custom env | Low | High | Test with random agent before RL training |
| Week 3 experiments take too long to train | Medium | Medium | Run on smaller grid, parallelize where possible |

---

## Team Roles
*(Fill in)*

| Member | Primary Responsibility |
|---|---|
| | Environment + fire spread module |
| | DQN training + tuning |
| | PPO training + tuning |
| | Visualization + analysis |
| | Report + presentation |

---

## Key Dates

| Date | Milestone |
|---|---|
| March 24 | Proposal submitted |
| March 30 | Environment + baseline complete |
| April 6 | DQN + PPO training complete |
| April 13 | Experiments + analysis complete |
| April 20 | Final report + presentation submitted |

---

*Last updated: March 23, 2026*
