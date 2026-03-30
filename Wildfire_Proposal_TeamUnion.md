**MMAI-845 — Reinforcement Learning**

**Team Union — Project Proposal**

**Autonomous Firefighting Resource Routing via Reinforcement Learning**

---

**Problem Description**

Wildfire containment is a time-critical resource routing problem. An incident commander must route limited suppression assets — aerial tankers, hotshot crews, and bulldozers — to build containment lines before a spreading fire reaches populated zones. Burned terrain is permanently impassable, creating a continuously shrinking action space that distinguishes this from standard routing problems and makes it a compelling RL domain.

We extend the Maze environment with a custom fire spread module using a cellular automaton. Each timestep, fire propagates to adjacent cells based on wind direction; burned cells become permanent walls.

**State & Action Space**

| Variable | Description |
|---|---|
| Agent position | (x, y) on terrain grid |
| Fire spread map | Unburned / on fire / burned / containment line |
| Wind direction | One-hot: N/S/E/W |
| Resources remaining | Suppression capacity left |
| Populated zones | Fixed high-value cells to protect |
| Time elapsed | Urgency signal |

**Actions (6 discrete):** Move N/S/E/W · Deploy containment line · Hold position.

The agent is rewarded for containment placements, penalized per timestep, and receives a large negative reward if fire reaches a populated zone. We compare **DQN** (off-policy, replay buffer suited to non-stationary dynamics) and **PPO** (on-policy, stable in dynamic environments) against a random agent and a greedy shortest-path baseline, evaluating on containment success rate, response speed, and resource efficiency across held-out scenarios initialized from real BC Wildfire Service historical fire perimeters.

---

**Business Case**

BC Wildfire Service responded to 2,200+ fires in 2023, burning 2.84M hectares at $720M in suppression costs. Incident commanders make real-time routing decisions under extreme cognitive load with incomplete situational awareness. This project builds a proof-of-concept RL routing recommendation engine functioning as a decision support layer for fire incident command. Beyond public agencies (BC Wildfire Service, Parks Canada, USDA Forest Service), the framework addresses catastrophe insurers (Swiss Re, Munich Re) seeking higher-fidelity containment models for loss estimation, and autonomous aerial firefighting platforms (Joby Aviation, Natilus) requiring onboard policy layers for uncrewed suppression missions.
