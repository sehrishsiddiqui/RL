# MMAI-845 — Project Proposals
### 3 Draft Proposals for Team Review

---

## PROPOSAL 1 — Wildfire Containment & Firefighting Resource Routing

**Team XXXXX — Project Proposal**

**Autonomous Firefighting Resource Routing via Reinforcement Learning**

---

**Problem Description**

Wildfire containment is a time-critical, spatially constrained resource routing problem. A fire incident commander must route limited suppression resources — aerial tankers, hotshot crews, bulldozers — through terrain to establish containment lines before a spreading fire reaches populated zones. Critically, every timestep of inaction allows the fire front to expand, permanently closing off routing options and escalating the risk of catastrophic loss. The irreversibility of burned terrain is what distinguishes this from a standard routing problem and makes it a compelling RL domain.

We will use the Maze environment as our terrain simulation, extended with a custom dynamic wall generation module implementing fire spread via a cellular automaton. Each timestep, fire propagates to adjacent unburned cells based on wind direction and terrain slope — burned cells become permanently impassable walls, creating a continuously shrinking action space. The state space encodes agent position, the current fire spread map, wind direction vector, remaining suppression resources, and distance of active fire front from designated protected zones. The discrete action space includes four directional moves, a deploy-resource action (establish containment line at current cell), and a hold action to conserve resources. We compare DQN and PPO: DQN's off-policy replay buffer may better handle the non-stationary environment dynamics, while PPO's on-policy stability may produce more consistent containment strategies. A greedy shortest-path agent serves as the non-learning baseline.

---

**Business Plan / Product Link**

BC Wildfire Service responded to over 2,200 wildfires in 2023, burning 2.84 million hectares — the most destructive season in provincial history, costing an estimated $720 million in suppression costs alone. Incident commanders make real-time routing decisions under extreme cognitive load, with incomplete situational awareness, often coordinating dozens of aircraft and ground crews simultaneously across hundreds of kilometres of active fire perimeter. Routing errors — pre-positioning crews in sectors that burn before they arrive, or failing to anticipate wind-driven spotting events — translate directly into containment failures, community evacuations, and loss of life.

This project develops a proof-of-concept RL decision support system for fire incident command. The agent is not designed to replace commanders but to function as a routing recommendation engine: given a live fire spread forecast and available resource inventory, the agent proposes an optimal deployment sequence. This is analogous to how air traffic control uses algorithmic routing assistance alongside human controllers. The immediate commercial customer is provincial and federal wildfire agencies (BC Wildfire Service, Parks Canada, USDA Forest Service), which have all publicly committed to technology modernization investments following the 2023 season.

The commercial opportunity extends significantly into the catastrophe insurance industry. Swiss Re, Munich Re, and FM Global collectively underwrite billions in wildfire-exposed commercial property across Western North America. Their catastrophe models price insurance premiums using fire spread simulations — but these simulations currently assume static or heuristic containment responses. An RL agent that learns realistic, adaptive containment routing policies provides a higher-fidelity simulation of human incident response, directly improving the accuracy of loss estimates used for premium pricing and reinsurance structuring. A third application exists in the emerging autonomous aerial firefighting space: Joby Aviation, Natilus, and several DARPA-funded programs are developing uncrewed aerial vehicles for fire suppression, where the learned routing policy maps directly to autonomous flight path planning. The RL framework developed here represents the policy layer that would govern such a system.

---

## PROPOSAL 2 — Organ Transplant Logistics Optimization

**Team XXXXX — Project Proposal**

**Time-Critical Organ Transport Routing via Reinforcement Learning**

---

**Problem Description**

Organ transplant logistics is among the most time-sensitive sequential decision problems in existence. Once procured, a donated heart remains viable for 4–6 hours; a liver for 12–24 hours; a kidney for 24–36 hours. A logistics coordinator must route transport vehicles through a hospital network — from donor procurement site to recipient transplant centre — before the organ's biological clock expires. Unlike conventional routing problems where the objective is purely to minimize distance or cost, here the reward degrades continuously with elapsed time, creating a fundamentally different optimization landscape where speed dominates efficiency.

We will use the Pick-up and Delivery environment from the OR Library, which provides a graph-based hospital network with discrete routing actions compatible with Stable-Baselines3. The environment is extended with a time-decay reward shaping mechanism: reward at delivery is defined as R × exp(−λ × t), where t is elapsed timesteps since procurement and λ is an organ-specific decay constant (heart: λ = 0.30, liver: λ = 0.12, kidney: λ = 0.05). This forces the agent to internalize urgency as a primary policy signal rather than a soft constraint. The state space encodes current vehicle location, organ type and remaining viability window, recipient hospital node, and graph edge weights representing transport duration. The discrete action space covers graph traversal (move to adjacent hospital node), pick-up, and deliver. We compare PPO and A2C, with a greedy shortest-path algorithm as baseline, and systematically analyze how the λ parameter shapes learned policy behavior: does aggressive decay produce shorter but riskier routes that sacrifice route quality for raw speed?

---

**Business Plan / Product Link**

The United Network for Organ Sharing (UNOS) coordinates over 45,000 transplants annually across 250+ transplant centres in the United States. Despite this scale, the organ discard rate remains alarmingly high: approximately 20% of procured kidneys and 8% of procured livers are discarded before transplant, with logistical failure — not medical unsuitability — identified as a primary contributing factor in peer-reviewed literature (Mohan et al., JAMA Surgery, 2018). Organs are delayed or lost due to suboptimal routing decisions made by human coordinators under extreme time pressure, with incomplete real-time data on transport availability, weather, and surgical team readiness. Each discarded organ represents not only a preventable death but an estimated $1.5M in foregone transplant revenue and post-transplant care for the receiving hospital system.

This project addresses that coordination failure directly. The RL agent learns organ-type-specific routing policies that adapt to viability window constraints in real time — treating a heart transport with the urgency of an emergency scramble and a kidney transport with the optimization patience its longer window permits. The practical product is a routing recommendation module integrated into UNOS's existing DonorNet platform, which already serves as the electronic hub for donor-recipient matching and inter-centre communication. DonorNet processes real-time data on donor status, recipient waitlist priority, and transplant centre capacity — precisely the state information our agent requires. The integration path is therefore an API-layer addition, not a platform replacement, reducing adoption friction significantly.

The commercial opportunity extends into the air medical transport sector. Companies including Air Methods, PHI Air Medical, and Med-Trans collectively fly thousands of organ transport missions annually, billing transplant centres $15,000–$80,000 per flight. An RL-optimized routing layer that demonstrably reduces average transport time — even by 20 minutes per mission — produces measurable clinical outcome improvements (organ quality at implantation correlates directly with cold ischaemia time) and reduces the per-organ transport cost by enabling more efficient multi-leg routing. For transplant centres operating under CMS quality metrics and UNOS performance benchmarks, a reduction in organ discard rate attributable to an RL logistics system is a quantifiable, auditable quality improvement with direct reimbursement and accreditation implications. The ethical imperative — saving lives through better logistics — provides a compelling regulatory and public affairs narrative that accelerates institutional adoption.

---

## PROPOSAL 3 — Pharmaceutical Inventory Sensitivity Analysis

**Team XXXXX — Project Proposal**

**Robust Inventory Policy Learning Under Non-Stationary Demand: A Pharmaceutical Supply Chain Study**

---

**Problem Description**

Pharmaceutical inventory management is a sequential decision problem under uncertainty with asymmetric costs. A supply chain manager must decide daily order quantities for critical medications without knowing future demand. The cost structure is fundamentally different from consumer goods: a stockout of a chemotherapy drug or ICU sedative carries consequences — patient harm, regulatory scrutiny, liability — that are orders of magnitude larger than any holding cost. Classical inventory theory addresses this through the Economic Order Quantity (EOQ) formula and base stock policies, which derive analytically optimal solutions under a critical but frequently violated assumption: that demand follows a stationary distribution. This assumption fails precisely when it matters most — during disease outbreaks, drug recalls, biosimilar launches, and supply chain disruptions.

We will use the Inventory Control environment with a discrete state and action space, comparing five policy types: PPO, A2C, tabular Q-learning, EOQ, and the base stock policy. The core research contribution is a systematic sensitivity analysis structured around five experimental dimensions: (1) demand distribution shape — Poisson, Gaussian, and heavy-tailed to test robustness to distribution misspecification; (2) stockout penalty magnitude — sweeping from low to catastrophic to identify the cost threshold at which RL outperforms analytical solutions; (3) lead time length — testing how longer procurement delays affect policy quality; (4) non-stationary demand shock — a mid-episode 3× demand surge simulating a disease outbreak, measuring which policies adapt and which collapse; and (5) inventory capacity constraints — testing behavior under tight versus generous storage limits. The central research question is precise and answerable: under what combination of demand volatility and cost asymmetry does deep RL provide a statistically significant advantage over classical inventory methods — and what is the magnitude of that advantage?

---

**Business Plan / Product Link**

McKesson Corporation, North America's largest pharmaceutical distributor, manages inventory replenishment across 40,000+ pharmacy and hospital clients, processing $277 billion in annual revenue. Their inventory policies — largely rule-based reorder-point systems parameterized on historical demand averages — govern whether a hospital pharmacy maintains adequate stock of insulin, vancomycin, or fentanyl. The COVID-19 pandemic exposed the catastrophic fragility of these policies: ventilator shortages, PPE stockouts, and remdesivir allocation failures were all downstream consequences of inventory systems that had no mechanism to adapt when demand distributions shifted overnight. The FDA's drug shortage database currently tracks over 130 active shortages, a figure that has remained persistently elevated since 2020.

The product application of this research is an adaptive policy engine embedded within McKesson's existing ERP infrastructure (SAP S/4HANA, Oracle SCM). The system operates in two modes governed by a demand regime classifier: under stationary demand, it defers to the computationally efficient EOQ baseline — our sensitivity analysis will quantify precisely the conditions under which this is safe and optimal. When the classifier detects a distributional shift (flagged by anomaly detection on incoming order patterns), the system activates the pre-trained RL policy, which has been specifically optimized for non-stationary demand conditions. This hybrid architecture — classical policy under stability, RL policy under shock — is the direct commercial translation of our experimental findings and avoids the sample efficiency disadvantage of RL under normal operating conditions.

The regulatory dimension provides a second distinct commercial pathway. The FDA's Drug Shortage Staff and the HHS Office of the Assistant Secretary for Preparedness and Response (ASPR) have both issued guidance requiring pharmaceutical manufacturers to submit supply chain resilience assessments as part of drug application filings. An RL-based adaptive inventory system that can demonstrably outperform static policies during simulated demand shocks — backed by the sensitivity analysis results from this project — provides manufacturers with a quantifiable, reproducible resilience metric suitable for regulatory submission. For a mid-size specialty pharmaceutical manufacturer managing 50–200 SKUs of critical medications, the cost of a single drug shortage event (FDA fines, lost contracts, remediation) routinely exceeds $50M. The business case for a policy system that provably reduces shortage probability under demand shock conditions is immediate and financially material.

---

*All proposals follow the MMAI-845 Project Proposal Outline: Problem Description (30%) and Business Plan/Product Link (70%). For final submission, reformat to under 1 page in PDF or DOCX.*
