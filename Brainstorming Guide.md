# MMAI-845 — RL Team Project Brainstorming Guide
### 4 Candidate Ideas for Team Discussion

---

## IDEA 1 — EV Charging Station Energy Management

### Concept Summary
A charging network operator must decide how much energy to procure from the grid each time period to meet uncertain EV charging demand, while minimizing cost and avoiding unmet demand penalties. The inventory is energy reserves. The demand is EV fleet charging needs.

### Environment
- **Package:** Inventory Control (OR Library / custom Gym-compatible)
- **Type:** Single-agent, discrete state and action space
- **Task:** Simple Inventory Control with Lost Sales — adapted for energy reserves

### State Space
| Variable | Description |
|---|---|
| Current energy reserve | Units of energy stored (0 to max capacity) |
| Time of day | Hour bucket (peak vs off-peak pricing) |
| Day of week | Weekday vs weekend demand patterns |
| Vehicles queued | Number of EVs waiting to charge |
| Energy price signal | Current grid price (low / medium / high) |
| Weather indicator | Temperature proxy (cold = higher battery drain) |

### Action Space
- **Discrete:** Order quantity of energy units for next period
  - `0` = order nothing
  - `1` = order 25% of max capacity
  - `2` = order 50% of max capacity
  - `3` = order 75% of max capacity
  - `4` = order 100% of max capacity

### Reward Function
```
Reward = Revenue from charging sessions
       - Cost of energy procured (price-weighted)
       - Penalty for unmet demand (vehicles turned away)
       - Holding cost for stored energy (degradation)
```
Optional: Add a cliff penalty when reserve drops to zero (stranded fleet scenario)

### Algorithms to Compare
| Algorithm | Type | Rationale |
|---|---|---|
| PPO | Deep RL, on-policy | Strong general baseline |
| A2C | Deep RL, on-policy | Faster, good for discrete spaces |
| Tabular Q-learning | Classic RL | Baseline to show deep RL value |
| EOQ Formula | Analytical (non-RL) | Classical operations research baseline |

### Data Needed
- **For training:** None — environment generates stochastic demand internally
- **For business case enrichment (optional):**
  - Ontario IESO hourly electricity pricing (free at ieso.ca)
  - NREL EV charging demand datasets (free at nrel.gov)
  - ChargePoint public network utilization reports

### Key Research Question
> *Under which demand and pricing conditions does RL outperform classical EOQ policy, and where does it break down?*

### Experimental Design
1. Train PPO / A2C / Q-learning under stationary demand distribution
2. Introduce a **demand shock** mid-episode (fleet size doubles — EV adoption surge)
3. Compare policy adaptation: does RL recover? Does EOQ fail?
4. Sensitivity analysis: vary holding cost, stockout penalty, and demand variance
5. Plot: cost per charge session, unmet demand rate, reserve utilization

### Pros
- Fastest training of all four ideas — most time for analysis
- Four-way comparison (PPO, A2C, Q-table, EOQ) is academically thorough
- Demand shock / non-stationary test is original and practically meaningful
- EV is universally recognized, timely topic

### Cons
- Environment itself is simple — intellectual contribution lives in experimental design
- Less visually dramatic than wildfire or organ transplant
- Requires careful writing to distinguish from a basic inventory homework problem

### Business Case
**Company:** ChargePoint, Tesla Supercharger, Amazon Last-Mile Fleet, Hydro One EV Program
**Problem:** Energy procurement decisions are currently rule-based (reorder when below X%). RL learns a dynamic policy that responds to price signals, demand forecasts, and fleet composition — directly reducing procurement cost and improving fleet uptime.

---

## IDEA 2 — Wildfire Containment & Firefighting Resource Routing

### Concept Summary
A fire incident commander must route firefighting resources (air tankers, hotshot crews) through terrain to build containment lines before a spreading wildfire reaches populated areas. The maze is the terrain. Fire spreads dynamically, making cells impassable over time. The agent must intercept the fire front before it escapes.

### Environment
- **Package:** Maze environment (custom-modified)
- **Type:** Single-agent, discrete state and action space
- **Modification:** Dynamic wall generation using a cellular automaton to simulate fire spread

### Environment Modification Detail
```python
# Fire spreads each timestep based on wind direction + terrain slope
# Burned cells become impassable walls
# Agent must reach fire perimeter cells before they spread further
# Populated zone cells = catastrophic negative reward if fire reaches them
```

### State Space
| Variable | Description |
|---|---|
| Agent position | (x, y) coordinates on terrain grid |
| Fire spread map | Binary grid: burned / not burned |
| Wind direction | N / S / E / W vector |
| Resources remaining | Units of suppression capacity left |
| Populated zone locations | Fixed high-value cells to protect |
| Time elapsed | Urgency signal — fires grow exponentially |

### Action Space
- **Discrete:**
  - `0` = Move North
  - `1` = Move South
  - `2` = Move East
  - `3` = Move West
  - `4` = Deploy suppression resource (build containment line at current cell)
  - `5` = Wait / hold position (conserve resources)

### Reward Function
```
Reward = +50 per containment line cell successfully placed
       - 1  per timestep (urgency — encourage speed)
       - 100 if fire reaches any populated zone cell
       + 200 upon full containment (fire stopped)
       - 20  per resource unit wasted on already-burned cells
```

### Algorithms to Compare
| Algorithm | Type | Rationale |
|---|---|---|
| DQN | Deep RL, off-policy | Strong for discrete navigation tasks |
| PPO | Deep RL, on-policy | Better exploration in dynamic environments |

### Data Needed
- **For training:** None — maze + fire spread generated procedurally
- **For business case enrichment (optional):**
  - BC Wildfire Service historical fire perimeter data (open.canada.ca)
  - NASA FIRMS active fire map data (firms.modaps.eosdis.nasa.gov) — free
  - Terrain elevation data from SRTM (free, NASA)

### Key Research Question
> *Does an RL agent learn a proactive containment strategy (intercept fire front early) or a reactive one (chase the fire), and how does wind direction affect the learned policy?*

### Experimental Design
1. Train DQN and PPO on small static maze (no fire spread) — baseline
2. Introduce dynamic fire spread — compare adaptation
3. Vary wind direction and speed across episodes — test policy generalization
4. Compare: random agent, greedy shortest-path agent, DQN, PPO
5. Visualization: heatmap of agent positions overlaid on fire spread progression

### Pros
- Dynamic environment modification is technically impressive
- Fire spread visualization is the best presentation asset of all four ideas
- Real societal relevance — BC, California, Australia wildfire crises
- Novel enough that no other team will attempt it

### Cons
- Dynamic maze modification requires custom Gym environment wrapper
- Fire spread + agent training interaction can cause instability — needs careful tuning
- More setup time than EV or Inventory ideas

### Business Case
**Company:** BC Wildfire Service, CAL FIRE, US Forest Service, Descartes Labs (fire risk modeling), insurance underwriters (FM Global, Swiss Re)
**Problem:** Incident commanders make routing decisions manually under extreme cognitive load and time pressure. An RL agent trained on simulated terrain scenarios provides decision support — suggesting optimal resource routing given live fire spread predictions.

---

## IDEA 3 — Organ Transplant Logistics Optimization

### Concept Summary
Donated organs have hard biological viability windows. A heart must be transplanted within 4–6 hours of procurement; a kidney within 24–36 hours. The agent is a transplant logistics coordinator routing transport vehicles from donor hospitals to recipient hospitals across a network, racing against time. Delay = organ loss = maximum negative reward.

### Environment
- **Package:** OR Library — Pick-up & Delivery environment
- **Type:** Single-agent, discrete action space
- **Key modification:** Time-decay reward shaping (reward degrades as time steps increase)

### Environment Modification Detail
```python
# Organ viability window mapped to episode timestep budget
# Reward multiplier decays exponentially with time elapsed:
# reward_multiplier = exp(-lambda * time_elapsed)
# lambda tuned per organ type (heart: aggressive decay, kidney: gentle decay)
# Failed delivery beyond window = large negative terminal reward
```

### State Space
| Variable | Description |
|---|---|
| Current vehicle location | Node ID in hospital network graph |
| Organ type | Heart / kidney / liver / lung (different viability windows) |
| Time elapsed since procurement | Timesteps since organ was harvested |
| Remaining viability window | Hard deadline countdown |
| Recipient hospital location | Target node |
| Traffic/route conditions | Edge weights on the network graph |
| Vehicle capacity | Single organ vs multi-organ transport |

### Action Space
- **Discrete:**
  - `0–N` = Route to adjacent hospital node (graph traversal)
  - `N+1` = Pick up organ at current location
  - `N+2` = Deliver organ at current location

### Reward Function
```
Reward = +100 * exp(-λ * time_elapsed)   # successful delivery, time-decayed
       - 1    per timestep               # urgency penalty
       - 200  if organ expires in transit # biological timeout
       - 10   per unnecessary detour     # route inefficiency
       + 20   bonus for beating expected # faster than baseline ETA
              transport time
```
Where λ (decay rate) is organ-specific: heart=0.3, kidney=0.05, liver=0.1

### Algorithms to Compare
| Algorithm | Type | Rationale |
|---|---|---|
| PPO | Deep RL, on-policy | Handles reward shaping well |
| A2C | Deep RL, on-policy | Faster convergence, good baseline |

### Data Needed
- **For training:** None — OR Library generates network topology procedurally
- **For business case enrichment (optional):**
  - UNOS/OPTN public transplant outcome data (optn.transplant.hrsa.gov) — free
  - Canadian Blood Services organ transport statistics (blood.ca) — free
  - Academic: "Organ Transplant Logistics" literature (Google Scholar)

### Key Research Question
> *How does the time-decay reward shaping (λ parameter) affect agent behavior — does aggressive decay produce more urgent but less optimal routes?*

### Experimental Design
1. Train PPO and A2C without time-decay reward (baseline routing task)
2. Add time-decay reward — compare policy shift: does agent take riskier shortcuts?
3. Test across organ types (different λ values) — does agent adapt to urgency level?
4. Compare against greedy shortest-path baseline
5. Analysis: delivery success rate, average time-to-delivery, route efficiency score

### Pros
- Time-decay reward shaping is a sophisticated, publishable-quality concept
- Highest-stakes business case — literally life and death
- Memorable and emotionally resonant for any audience
- OR Library is Gym-compatible, Stable-baselines3 works directly

### Cons
- Reward shaping requires careful design to avoid reward hacking
- OR Library is less documented — initial setup takes more effort
- Must be careful not to oversell capabilities (real organ logistics is far more complex)

### Business Case
**Company:** UNOS (United Network for Organ Sharing), Canadian Blood Services, LifeSource, hospital network logistics departments
**Problem:** Organ coordinators currently make routing decisions manually, relying on experience and fixed protocols. Thousands of organs are discarded annually due to logistical failures — not medical unsuitability. An RL agent that learns time-optimal routing policies under viability constraints could directly reduce organ discard rates.

---

## IDEA 4 — Inventory Sensitivity Analysis (Pharmaceutical Supply Chain)

### Concept Summary
A pharmaceutical manufacturer must manage global inventory of critical medications under uncertain demand. The key research question is: *when does RL outperform classical inventory theory, and under which demand conditions does it fail?* This is a rigorous comparative study across demand distributions, cost structures, and policy types.

### Environment
- **Package:** Inventory Control (Gym-compatible)
- **Type:** Single-agent, discrete state and action space
- **Modification:** Implement non-stationary demand (mid-episode distribution shift to simulate a disease outbreak surge)

### State Space
| Variable | Description |
|---|---|
| Current inventory level | Units on hand (0 to max capacity) |
| Demand history (window) | Last N periods of realized demand |
| Lead time | Periods until ordered stock arrives |
| Current holding cost | Cost per unit per period |
| Stockout penalty | Cost per unit of unmet demand |
| Demand regime | Stationary / shock / recovery (for non-stationary experiments) |

### Action Space
- **Discrete:** Order quantity for next period
  - `0` = Do not order
  - `1–K` = Order K units (K up to max order size)

### Reward Function
```
Reward = Revenue from fulfilled demand
       - Holding cost * units on hand
       - Stockout penalty * units of unmet demand
       - Order cost * units ordered
```

### Algorithms to Compare
| Algorithm | Type | Rationale |
|---|---|---|
| PPO | Deep RL, on-policy | Primary RL agent |
| A2C | Deep RL, on-policy | Comparison RL agent |
| Tabular Q-learning | Classic RL | Shows value of function approximation |
| EOQ Formula | Analytical | Classical OR baseline |
| Base Stock Policy | Analytical | Industry-standard pharmaceutical baseline |

### Sensitivity Analysis Matrix
| Experiment | Variable Changed | Question |
|---|---|---|
| 1 | Demand distribution (Poisson → Gaussian → Heavy-tailed) | Does RL handle non-normal demand better? |
| 2 | Stockout penalty (low → high) | Does RL adapt to critical vs non-critical items? |
| 3 | Lead time (1 → 5 periods) | Does RL handle longer lead times better than EOQ? |
| 4 | Demand shock (mid-episode surge 3x normal) | Does RL adapt? Does EOQ collapse? |
| 5 | Max inventory capacity (tight → generous) | How does constraint severity affect policy quality? |

### Data Needed
- **For training:** None — environment generates demand stochastically
- **For parameterizing demand distributions (optional):**
  - FDA drug shortage database (accessdata.fda.gov) — free
  - WHO essential medicines demand data — free
  - Academic: Cachon & Terwiesch "Matching Supply with Demand" (textbook reference)

### Key Research Question
> *Under stationary demand, classical EOQ matches or beats RL. Under non-stationary demand shocks, does RL's adaptability provide a measurable advantage — and how large must the shock be before RL wins?*

### Experimental Design
1. Establish EOQ and Base Stock analytical solutions as gold-standard benchmarks
2. Train PPO, A2C, Q-table under stationary demand — compare to benchmarks
3. Introduce non-stationary demand shock — measure policy degradation per method
4. Sweep across sensitivity analysis matrix (5 experiments above)
5. Final output: a 2x2 grid — RL advantage vs (demand volatility, stockout penalty)

### Pros
- Five-way comparison is the most academically rigorous of all four ideas
- Non-stationary demand is an original research contribution
- Fastest training — all compute time goes to experiments
- Pharmaceutical framing (critical drug shortages) gives high stakes to a simple environment

### Cons
- Simple environment — depth entirely dependent on experimental design quality
- Least visually impressive — results are charts and tables, not animations
- Requires strong written analysis to carry the report

### Business Case
**Company:** Pfizer, Johnson & Johnson global supply chain, McKesson pharmaceutical distribution, hospital pharmacy networks
**Problem:** Pharmaceutical inventory policies are typically rule-based (reorder point + safety stock). These fail during demand shocks (COVID-19 PPE, Ozempic shortage). An RL agent that dynamically adapts ordering policies to detected demand regime shifts could prevent critical drug shortages while reducing excess inventory costs.

---

## Summary Comparison

| | EV Charging | Wildfire | Organ Transplant | Inventory Sensitivity |
|---|---|---|---|---|
| **Environment** | Inventory | Maze (modified) | OR Library | Inventory |
| **Primary Algorithm** | PPO vs A2C | DQN vs PPO | PPO vs A2C | PPO vs A2C vs Q-table |
| **Action Space** | Discrete | Discrete | Discrete | Discrete |
| **Data Required** | None | None | None | None |
| **Custom Env Work** | Low | High | Medium | Medium |
| **Training Speed** | Fast | Medium | Medium | Fast |
| **Visual Impact** | ★★★ | ★★★★★ | ★★★★ | ★★ |
| **Academic Depth** | ★★★★ | ★★★★ | ★★★★★ | ★★★★★ |
| **Originality** | ★★★★ | ★★★★★ | ★★★★★ | ★★★★ |
| **Business Case Strength** | ★★★★ | ★★★★★ | ★★★★★ | ★★★★ |
