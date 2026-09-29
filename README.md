# Battery-Aware Drone Delivery in Stochastic Wind Conditions Using Reinforcement Learning

**Course:** Reinforcement Learning | **Type:** Mini-Project (Part I: CO1-CO3, Part II: CO4-CO5)  
**Team:** Member 1 (ID), Member 2 (ID), Member 3 (ID)  
**Faculty:** _Name_ | **Academic Year:** _Year_  

---

## 1. Overview & Problem Definition (CO1)

This project models an autonomous delivery drone navigating an $8 \times 8$ grid world. The drone is tasked with picking up packages at a central depot $(0, 0)$ and delivering them to three distinct destinations ($D_1, D_2, D_3$), where destination $D_3$ has an urgent time-critical deadline. The drone has a limited battery capacity, consumes more energy when loaded with a package, encounters stochastic wind zones near the center of the grid, and must execute tactical detours to recharging stations ($C_1, C_2$) to survive and complete all deliveries.

The problem is solved using tabular Reinforcement Learning implemented strictly from scratch in **NumPy** and **Gymnasium**, without any high-level RL libraries (e.g. Stable-Baselines3). Both off-policy **Q-Learning** and on-policy **SARSA** agents are implemented, trained, evaluated, and demonstrated step-by-step.

```
       8x8 Grid Layout (Cartesian: x=0..7, y=0..7)
  +---+---+---+---+---+---+---+---+
7 | . | . | . | . | . | . | . |D1 |  <- Normal Drop-off (7, 7)
  +---+---+---+---+---+---+---+---+
6 | . | . |D3 | . | . | . | . | . |  <- URGENT Drop-off (2, 6)
  +---+---+---+---+---+---+---+---+
5 | . | . | . | . | . | . | . | . |
  +---+---+---+---+---+---+---+---+
4 |C1 | . | . | W | W | . | . | . |  <- Charger C1 (0, 4) & Wind Zone
  +---+---+---+---+---+---+---+---+
3 | . | . | . | W | W | . | . | . |  <- Wind Zone Cluster (3..4, 3..4)
  +---+---+---+---+---+---+---+---+
2 | . | . | . | . | . | . | . |D2 |  <- Normal Drop-off (7, 2)
  +---+---+---+---+---+---+---+---+
1 | . | . | . | . | . | . | . | . |
  +---+---+---+---+---+---+---+---+
0 | H | . | . | . | . | . |C2 | . |  <- Depot H (0, 0) & Charger C2 (6, 0)
  +---+---+---+---+---+---+---+---+
    0   1   2   3   4   5   6   7
```

---

## 2. Why Reinforcement Learning? (CO1)

- **Sequential Decisions:** Each movement decision impacts immediate spatial position, remaining battery, and time remaining for the urgent delivery, constraining all future actions.
- **Closed-Loop Interaction:** The drone operates inside an active environment, receiving immediate sensory and reward feedback after every action.
- **Evaluative Feedback, Not Labels:** There is no supervised dataset of "optimal flight paths" under stochastic wind perturbations. The policy emerges purely by optimizing cumulative discounted returns.
- **Environmental Uncertainty:** In wind zones, transitions are stochastic (20% perpendicular deflection), requiring the agent to learn robust policies that succeed in expectation.

### Comparison: Supervised Learning vs. Reinforcement Learning

| Aspect | Supervised / Conventional ML | Reinforcement Learning (This Project) |
|---|---|---|
| **Learning Signal** | Ground-truth labels or continuous targets | Scalar reward signal ($r_t$) reflecting task success and penalties |
| **Interaction** | Passive, static dataset; no action loop | Active closed-loop agent–environment interaction |
| **Decision** | One-shot static prediction | Sequential, multi-stage action selection |
| **Temporal Structure** | Independent and identically distributed (i.i.d.) | Non-i.i.d., trajectory-based Markov decision process |
| **Objective** | Minimize loss function $\mathcal{L}(y, \hat{y})$ | Maximize expected discounted return $\mathbb{E}\left[\sum_{t} \gamma^t r_t\right]$ |

---

## 3. Markov Decision Process (MDP) Formulation (CO2)

The environment is formalized as a Markov Decision Process: $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$.

### State Space ($\mathcal{S}$)
The state is represented as a 6-tuple and encoded bijectively into a single discrete integer index:
$$s = (x, y, \text{battery}, \text{carrying}, \text{delivered\_mask}, \text{urgent\_bucket})$$

| Variable | Description | Range / Values | Cardinality |
|---|---|---|:---:|
| $x$ | Horizontal grid coordinate | $0 \dots 7$ | 8 |
| $y$ | Vertical grid coordinate | $0 \dots 7$ | 8 |
| $\text{battery}$ | Discrete battery level | $0 \dots 30$ | 31 |
| $\text{carrying}$ | Flag: whether drone currently carries a package | $0$ (empty), $1$ (loaded) | 2 |
| $\text{delivered\_mask}$ | 3-bit binary mask of completed deliveries ($D_1, D_2, D_3$) | $0 \dots 7$ (`0b000` to `0b111`) | 8 |
| $\text{urgent\_bucket}$ | Coarse timer bucket for urgent package $D_3$ | 0 (Safe), 1 (Warning), 2 (Expired) | 3 |

**Total Discrete States:**
$$|\mathcal{S}| = 8 \times 8 \times 31 \times 2 \times 8 \times 3 = 95,232 \text{ states}$$

Bijective mixed-radix encoding ensures instant $O(1)$ indexing into NumPy arrays:
$$\text{Index} = (((((x \cdot 8 + y) \cdot 31 + \text{battery}) \cdot 2 + \text{carrying}) \cdot 8 + \text{delivered\_mask}) \cdot 3 + \text{urgent\_bucket}$$

### Action Space ($\mathcal{A}$)
Discrete set of 6 primitive decisions ($|\mathcal{A}| = 6$):

| ID | Action | Dynamics & Mechanics |
|:---:|---|---|
| **0** | `UP` | Move one cell North ($y \leftarrow \min(7, y+1)$). Drains battery. |
| **1** | `DOWN` | Move one cell South ($y \leftarrow \max(0, y-1)$). Drains battery. |
| **2** | `LEFT` | Move one cell West ($x \leftarrow \max(0, x-1)$). Drains battery. |
| **3** | `RIGHT` | Move one cell East ($x \leftarrow \min(7, x+1)$). Drains battery. |
| **4** | `PICKUP/DROPOFF` | If at Depot and empty, loads package. If at undelivered $D_i$ with package, delivers it. Invalid elsewhere. |
| **5** | `CHARGE` | Restores $+2$ battery per step if at charger ($C_1$ or $C_2$) and $\text{battery} < 30$. Invalid elsewhere. |

### Transition Dynamics ($\mathcal{P}$)
- **Movement:** Moving off the $8 \times 8$ grid is blocked (drone stays in place).
- **Stochastic Wind:** If the drone moves into a wind cell $(x, y) \in \{(3,3), (3,4), (4,3), (4,4)\}$, with probability $p = 0.20$ it experiences an unexpected sideways deflection perpendicular to its intended heading.
- **Battery Consumption:** Flying loaded consumes 2 units/step; flying empty or hovering consumes 1 unit/step.
- **Recharging:** Staying at a charger and choosing `CHARGE` restores $+2$ battery per step until full ($30$).
- **Termination:** Episode terminates if all 3 packages are delivered (success) or battery reaches 0 (failure). Truncated if steps reach $200$.

### Reward Function ($\mathcal{R}$) & Justification

| Event | Reward | Justification / Viva Explanation |
|---|:---:|---|
| **Normal Delivery ($D_1, D_2$)** | `+50.0` | Substantial milestone reward for completing delivery legs. |
| **Urgent Delivery ($D_3$) on Time** | `+100.0` | Maximum positive incentive for satisfying the tight deadline ($\le 60$ steps). |
| **Urgent Delivery ($D_3$) Late** | `+20.0` | Diminished return for overdue delivery. |
| **Step Penalty** | `-1.0` | Incurred every step; encourages shortest, most energy-efficient paths. |
| **Urgent Deadline Missed** | `-30.0` | One-off penalty triggered at step 60 if $D_3$ has not been delivered. |
| **Battery Exhaustion (Death)** | `-100.0` | Catastrophic terminal penalty when battery hits 0; terminates episode. |
| **Useless Action** | `-2.0` | Penalizes invalid actions (charging away from stations, dropping off in empty cells). |
| **Pickup Subgoal Bonus** | `+20.0` | Intermediate reward when loading a needed package at Depot. Prevents premature termination and provides a gradient for the drone to return home after delivery. |
| **Recharge Step Incentive** | `+1.2` | With `-1.0` step penalty, net reward is `+0.2` while battery $< 30$. This crucial tuning incentivizes the agent to remain at the charger until completely full before embarking on long flights. |

---

## 4. Algorithms Implemented (CO3)

### 1. Tabular Q-Learning (Off-Policy TD Control)
Learns the optimal action-value function $Q^*$ directly, updating towards the greedy action at $s'$:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

### 2. Tabular SARSA (On-Policy TD Control)
Updates action values using the action $a'$ actually selected by the behavior policy ($\epsilon$-greedy) at $s'$:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

Both agents inherit from [`BaseAgent`](base_agent.py), ensuring clean DRY code where the difference between on-policy and off-policy TD learning is visible at a glance.

---

## 5. Experimental Results (CO3, Part I Benchmark)

Evaluation over 500 greedy evaluation episodes ($\epsilon = 0.0$):

| Metric | Random Baseline | Q-Learning (Off-Policy) | SARSA (On-Policy) | Viva Interpretation |
|---|:---:|:---:|:---:|---|
| **3-Package Success Rate** | **0.0%** | **100.0%** | **99.6%** | Both agents reliably master all 3 deliveries with detours. |
| **Mean Cumulative Return** | $-126.34 \pm 18.77$ | **$+209.00 \pm 0.00$** | **$+202.70 \pm 10.01$** | Clear transition from failure to high positive return. |
| **Average Deliveries Completed** | $0.00$ / 3.0 | **$3.00$ / 3.0** | **$3.00$ / 3.0** | Agent delivers all 3 packages every episode. |
| **Urgent Delivery ($D_3$) On-Time** | $0.0\%$ | **100.0%** | **100.0%** | Urgent package delivered well within the 60-step limit. |
| **Average Steps Taken** | 24.2 (died) | **87.0 steps** | **89.9 steps** | Q-learning finds the slightly shorter, optimal trajectory. |
| **Battery Exhaustion Rate** | **100.0%** | **0.0%** | **0.4%** | Charging detours are successfully planned and executed. |
| **Training Time (30,000 ep)** | N/A | **17.97 s** | **13.38 s** | Highly efficient pure-NumPy tabular execution. |

---

## 6. Project Structure

```
rl_mini_project/
|-- config.py              # Centralized layout, reward values, and hyperparameters
|-- drone_env.py           # Gymnasium environment, bijective encoder, text renderer, random baseline
|-- base_agent.py          # Abstract base class: Q-table storage, epsilon-greedy action selection
|-- q_agent.py             # Q-Learning agent with off-policy Bellman update
|-- sarsa_agent.py         # SARSA agent with on-policy SARSA update
|-- train.py               # Training loop, periodic progress logging, CSV logger, curve plotting
|-- evaluate.py            # 500-episode greedy evaluation suite with detailed statistics
|-- demo.py                # Step-by-step interactive live demo showing state -> action -> reward -> update
|-- requirements.txt       # Dependencies: gymnasium, numpy, matplotlib, pandas
|-- README.md              # Project documentation and Part I viva preparation guide
`-- results/
    |-- q_table_qlearning.npy          # Trained Q-Learning policy table
    |-- q_table_sarsa.npy              # Trained SARSA policy table
    |-- training_log_qlearning.csv     # Episodic training logs (30,000 episodes)
    |-- training_log_sarsa.csv         # Episodic training logs (30,000 episodes)
    `-- plots/
        |-- learning_curve_qlearning.png   # Learning curve: Return, success rate, and epsilon
        `-- learning_curve_sarsa.png       # Learning curve: Return, success rate, and epsilon
```

---

## 7. Setup & Execution Guide

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Verify Environment & Random Baseline
```bash
python drone_env.py
```

### 3. Train the Agents
```bash
# Train Q-Learning agent (takes ~18 seconds)
python train.py --algo qlearning

# Train SARSA agent (takes ~14 seconds)
python train.py --algo sarsa
```

### 4. Evaluate Trained Policies
```bash
# Evaluate Q-Learning (500 episodes, greedy)
python evaluate.py --algo qlearning

# Evaluate SARSA (500 episodes, greedy)
python evaluate.py --algo sarsa
```

### 5. Run Live Step-by-Step Demo
```bash
# Interactive mode (press Enter to step through):
python demo.py --algo qlearning

# Automated playback (fast mode):
python demo.py --algo qlearning --fast
```

---

## 8. Key Hyperparameters Used

| Parameter | Value | Justification |
|---|:---:|---|
| **Learning Rate ($\alpha$)** | `0.15` | Fast TD convergence without parameter oscillation. |
| **Discount Factor ($\gamma$)** | `0.98` | Encourages long-horizon planning across multi-stage deliveries. |
| **Initial Epsilon ($\epsilon_{\text{start}}$)** | `1.0` | Comprehensive early exploration of the 95,232-state space. |
| **Minimum Epsilon ($\epsilon_{\text{min}}$)** | `0.03` | Ensures asymptotic convergence while maintaining minor exploratory robustness. |
| **Epsilon Decay Rate** | `0.99985` | Smooth multiplicative decay across 30,000 episodes. |
| **Max Steps per Episode** | `200` | Ample time for 3 deliveries and 2 recharging detours (~87 steps needed). |
| **Random Seed** | `42` | Strictly deterministic experiments for full reproducibility. |

---

## 9. Part I Evidence Map (CO1-CO3)

| Assessment Requirement | Where to Find It in Codebase |
|---|---|
| **RL Justification (CO1)** | Section 2 above; [README.md](README.md) |
| **Problem Formulation: MDP (CO2)** | Section 3 above; [`config.py`](config.py), [`drone_env.py`](drone_env.py) |
| **Bijective State Construction** | [`drone_env.py`](drone_env.py) (`encode_state`, `decode_state`) |
| **Action Space & Dynamics** | [`drone_env.py`](drone_env.py) (`step`) |
| **Q-Learning Implementation** | [`q_agent.py`](q_agent.py) (`update`) |
| **SARSA Implementation** | [`sarsa_agent.py`](sarsa_agent.py) (`update`) |
| **Training Pipeline & Logging** | [`train.py`](train.py), [`results/training_log_qlearning.csv`](results/training_log_qlearning.csv) |
| **Trained Value Functions** | [`results/q_table_qlearning.npy`](results/q_table_qlearning.npy), [`results/q_table_sarsa.npy`](results/q_table_sarsa.npy) |
| **Learning Curves** | [`results/plots/learning_curve_qlearning.png`](results/plots/learning_curve_qlearning.png), [`results/plots/learning_curve_sarsa.png`](results/plots/learning_curve_sarsa.png) |
| **Step-by-Step Live Demo** | [`demo.py`](demo.py) (implements the full 8-step assessment sequence) |
| **Quantitative Policy Evaluation** | [`evaluate.py`](evaluate.py) |

---

## 10. Team Technical Responsibilities (3-Member Split)

This project is partitioned into three distinct technical specializations matching the evaluation rubric (Section 10 & 11 of the assessment guidelines). Each member has clear file ownership, technical deliverables, and specific viva defense questions:

```
+-----------------------------------------------------------------------------------+
|                                  THE 3-MEMBER SPLIT                               |
+--------------------------+------------------------------+-------------------------+
| Member 1: Environment &  | Member 2: RL Algorithms &    | Member 3: Evaluation,   |
| MDP Formulation          | Training Pipeline            | Demo & Policy Benchmarks|
| (CO1 & CO2 Focus)        | (CO2 & CO3 Focus)            | (CO3 & Part II Prep)    |
+--------------------------+------------------------------+-------------------------+
| * config.py              | * base_agent.py              | * demo.py               |
| * drone_env.py           | * q_agent.py                 | * evaluate.py           |
|                          | * sarsa_agent.py             | * results/plots/        |
|                          | * train.py                   |                         |
+--------------------------+------------------------------+-------------------------+
```

### Member 1: Environment Architect & MDP Formulation (CO1 & CO2)
* **Primary Code Files:** [`config.py`](config.py), [`drone_env.py`](drone_env.py)
* **Assigned Technical Deliverables:**
  - **Custom Gymnasium Environment:** Built [`DroneDeliveryEnv`](drone_env.py) subclassing `gymnasium.Env` with standard `reset()` and `step()` interfaces.
  - **MDP Formulation:** Formulated state space $\mathcal{S}$, action space $\mathcal{A}$, stochastic dynamics $\mathcal{P}$, and reward rules $\mathcal{R}$.
  - **State Encoding/Decoding:** Designed mixed-radix bijective mapping (`encode_state`, `decode_state`) mapping 6 variables into 95,232 discrete states ($O(1)$ indexing).
  - **Physical Dynamics:** Implemented payload-dependent battery drainage (empty vs loaded) and 20% perpendicular wind perturbation in central zones.
  - **Baseline Benchmark:** Created `random_policy_baseline()` demonstrating that random walk fails ($0\%$ success, $-126.34$ return, $100\%$ battery crash).
* **Viva Defense Questions & Talking Points:**
  - *Why is the problem an MDP?* Current state $(x, y, \text{battery}, \text{carrying}, \text{delivered}, \text{urgent\_bucket})$ satisfies the Markov property: $P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_t, a_t, s_{t-1}, \dots)$.
  - *Why bucket the urgent timer?* Discretizing remaining time into 3 coarse buckets (Safe, Warning, Expired) keeps the state space compact ($|\mathcal{S}| = 95,232$) without violating Markovian assumptions.

---

### Member 2: RL Algorithms & Training Pipeline (CO2 & CO3)
* **Primary Code Files:** [`base_agent.py`](base_agent.py), [`q_agent.py`](q_agent.py), [`sarsa_agent.py`](sarsa_agent.py), [`train.py`](train.py)
* **Assigned Technical Deliverables:**
  - **NumPy Agents From Scratch:** Built tabular RL foundation without third-party RL libraries.
  - **Off-Policy Q-Learning:** Implemented Bellman optimality updates targeting $\max_{a'} Q(s', a')$ in [`QLearningAgent`](q_agent.py).
  - **On-Policy SARSA:** Implemented on-policy updates targeting $Q(s', a')$ where $a'$ is sampled from behavior policy in [`SARSAAgent`](sarsa_agent.py).
  - **Exploration Scheduling:** Designed $\epsilon$-greedy action selection and smooth multiplicative decay ($\epsilon: 1.0 \to 0.03$) across 30,000 episodes.
  - **Reward Engineering:** Diagnosed exploration bottlenecks in multi-stage deliveries; added `REWARD_PICKUP` (+20) and `REWARD_CHARGE_STEP` (+1.2) to guide long-horizon convergence.
* **Viva Defense Questions & Talking Points:**
  - *Difference between Q-learning and SARSA in code?* Q-learning uses $\max_{a'} Q(s', a')$ (off-policy target), whereas SARSA uses $Q(s', a')$ chosen by the current policy (on-policy target).
  - *Why did reward tuning solve the learning deadlock?* Sparse terminal rewards failed because the drone exhausted its battery before random walks discovered subsequent drop-offs. The pickup bonus created an incentive to return home, while the net $+0.2$ charge incentive motivated the drone to stay at chargers until completely full.

---

### Member 3: Evaluation, Demonstration & Policy Benchmarking (CO3 & Part II Prep)
* **Primary Code Files:** [`demo.py`](demo.py), [`evaluate.py`](evaluate.py), [`results/plots/`](results/plots/)
* **Assigned Technical Deliverables:**
  - **Interactive Live Demo:** Created [`demo.py`](demo.py) executing the required 8-step viva sequence:
    $$\text{State} \to \text{Actions} \to \text{Decision} \to \text{Environment Response} \to \text{Reward} \to \text{TD Update} \to \text{Outcome}$$
  - **Statistical Policy Evaluator:** Built [`evaluate.py`](evaluate.py) evaluating 500 greedy episodes ($\epsilon = 0.0$) to record cumulative return, success rate, on-time urgent rate, steps, and battery health.
  - **Convergence Curves:** Automated generation of dual-panel plots in `results/plots/` illustrating return moving averages, 3-package completion rates, and $\epsilon$ decay.
  - **Part II Extensibility:** Structured logs (`results/training_log_<algo>.csv`) for Part II regret, sample complexity, and sensitivity analysis.
* **Viva Defense Questions & Talking Points:**
  - *What does the live demo prove?* Demonstrates every component in real-time: state decoding, greedy selection, environment transition, and explicit calculation of the Bellman TD target and TD error.
  - *What quantitative evidence proves convergence?* Q-Learning and SARSA achieved 100% 3-package delivery success rate, 100% on-time urgent delivery, and 0% battery exhaustion over 500 greedy evaluation episodes, compared to 0% success for the random baseline.

