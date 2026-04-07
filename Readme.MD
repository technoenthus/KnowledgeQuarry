# Learning Under Constraints
### KnowledgeQuarry · Problem 02 · ML Engineering Track
**Convoke 8.0 · Cluster Innovation Centre, University of Delhi**

---

## Problem Summary

Design a system where heterogeneous agents complete distributed tasks under resource constraints, and the system must **learn to improve its strategy over repeated episodes** — not just find a static solution.

---

## Our Approach

We model a **10×10 warehouse delivery grid** where two types of agents must pick up tasks scattered across the grid and deliver them to a central depot. A **Q-Learning agent** learns the optimal dispatch policy over 600 training episodes, demonstrating measurable improvement over time.

### Why Q-Learning?
- Tabular Q-learning is interpretable — judges can see exactly what the agent is learning
- Epsilon-greedy exploration gives a clean, visible learning curve
- Fast to train (~2 minutes), easy to reproduce

---

## Environment

| Parameter | Value |
|---|---|
| Grid size | 10 × 10 |
| Tasks per episode | 8 (randomly spawned) |
| Depot location | (5, 5) — centre |
| Max steps per episode | 200 |
| Episodes trained | 600 |

---

## Agent Specifications

| Property | Drone | Ground Robot |
|---|---|---|
| Speed | 2 cells/step | 1 cell/step |
| Task capacity | 3 tasks | 6 tasks |
| Energy limit | 40 steps | 120 steps |
| Starting position | (0, 0) | (9, 9) |
| Best suited for | Fast nearby pickups | Bulk long-range runs |

---

## Q-Learning Setup

| Hyperparameter | Value |
|---|---|
| Learning rate (α) | 0.15 |
| Discount factor (γ) | 0.95 |
| Epsilon start | 1.0 (fully random) |
| Epsilon min | 0.05 |
| Epsilon decay | 0.992 per episode |

**State space** (discretised tuple):
`(tasks_remaining, drone_energy%, robot_energy%, drone_cargo, robot_cargo, drone_dist, robot_dist)`

**Action space** (4 actions):
| Action | Description |
|---|---|
| 0 | Activate drone → pick up nearest task |
| 1 | Activate robot → pick up nearest task |
| 2 | Activate drone → return to depot |
| 3 | Activate robot → return to depot |

**Reward function:**
| Event | Reward |
|---|---|
| Task delivered to depot | +15 per task |
| Task picked up | +3 |
| Each movement step | −0.2 |
| Agent exhausted with cargo | −25 |

---

## Results

| Method | Mean Tasks/Episode |
|---|---|
| Random baseline | 1.57 |
| Greedy nearest-task | 1.01 |
| **Q-agent (last 100 eps)** | **1.91** |
| **Improvement vs random** | **+21.9%** |

The learning curve shows a clear upward trend from episode 1 onwards, with the Q-agent consistently outperforming both baselines after ~100 episodes.

---

## File Structure

```
├── kq_solution.py          # Training script — environment + Q-agent + baselines + plots
├── kq_dashboard.py         # Streamlit dashboard (run after training)
├── training_results.json   # Generated after training (auto-loaded by dashboard)
├── kq_results.png          # Learning curve + comparison plots (auto-generated)
└── README.md
```

---

## How to Run

**1. Install dependencies**
```bash
pip install numpy matplotlib streamlit plotly
```

**2. Train the Q-agent** (~2-3 minutes)
```bash
python kq_solution.py
```
This will print progress every 100 episodes, save `training_results.json` and `kq_results.png`, and display the results plot.

**3. Launch the interactive dashboard**
```bash
streamlit run kq_dashboard.py
```
Opens in your browser automatically. Shows learning curve, reward chart, epsilon decay, baseline comparison, and a live grid visualisation.

---

## Tech Stack

- **Python 3.8+**
- **NumPy** — environment simulation, Q-table operations
- **Matplotlib** — training plots
- **Streamlit** — interactive dashboard
- **Plotly** — dashboard charts and grid visualisation

---

## Team

Submitted for KnowledgeQuarry · Convoke 8.0 · Cluster Innovation Centre, University of Delhi