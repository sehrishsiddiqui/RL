# MMAI-845 — RL Team Project Brainstorming Guide
### Project Proposal Due: March 24

---

## How to Use This Document
Use this template to structure your project ideas before our meeting on **March 21st**.
For each idea you propose, fill out the sections below. Add as many ideas as you like — one table per idea.
We will review all submissions together on the 21st and finalize one idea to develop into the formal proposal.

---

## Quick Reference: Allowed Environments

| Environment | Type | Action Space | Notes |
|---|---|---|---|
| Inventory Control | Single-agent | Discrete | Day-to-day stock ordering under stochastic demand |
| Maze | Single-agent | Discrete | 2D navigation, supports portals and loops |
| AnyTrading | Single-agent | Discrete | Stock / ForEx trading with historical data |
| TensorTrade | Single-agent | Discrete/Continuous | More complex trading environment |
| OpenAI Gym Classic Control | Single-agent | Discrete | At least 2 environments required if chosen |
| OpenAI Gym Box2D | Single-agent | Continuous | Bipedal robot, racecar, lunar lander — 1 env needed |
| OR Library | Single-agent | Discrete | Traveling salesman, VRP, newsvendor, pick-up & delivery |

**Recommended learning package:** Stable-Baselines3 (compatible with all Gym environments)
Make sure the algorithm you select matches your environment's action space (discrete vs. continuous).

---

## Proposal Requirements (for reference)
The final proposal must be **under 1 page** and cover two sections:
- **Problem Description (30%):** Environment, algorithm, state space, action space, reward function
- **Business Plan / Product Link (70%):** How does this RL problem connect to a real business or product?

---

## Idea Template
*Copy and fill out one block per idea*

---

### IDEA — [Your Title Here]

**Concept Summary**
> What is the agent doing? What problem is it solving? (2–3 sentences)

**Environment**
- Package:
- Task:

**Algorithm(s)**
- Primary:
- Comparison (optional):

**State Space**
| Variable | Description |
|---|---|
| | |
| | |

**Action Space**
- List the discrete or continuous actions the agent can take

**Reward Function**
- What does the agent get rewarded for?
- What does it get penalized for?

**Data Needed**
- External data required (if any):
- Or: self-contained / procedurally generated

**Business Case**
- What company or industry does this apply to?
- What problem does it solve?
- Why is RL the right approach here?

**Pros**
-
-

**Cons**
-
-

---
*(Copy the block above for each additional idea)*
---
