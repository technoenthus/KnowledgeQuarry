"""
KnowledgeQuarry - Problem 02: Learning Under Constraints
Solution: Q-Learning for Multi-Agent Task Allocation

Environment:
  - Grid of N tasks. Each task has a location and a reward value.
  - Two agent types:
      Drone  : fast (speed=2), low capacity (3 tasks), short battery (40 steps)
      Robot  : slow (speed=1), high capacity (6 tasks), long battery (120 steps)
  - The Q-agent decides each step: which agent to activate and what to do.
  - Reward is dense and immediate so Q-learning can actually converge.

Run:
  pip install numpy matplotlib streamlit plotly
  python kq_solution.py
  streamlit run kq_dashboard.py
"""

import numpy as np
import random
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────

GRID     = 10
N_TASKS  = 8
N_EPS    = 600
MAX_STEPS= 200
DEPOT    = (5, 5)

AGENT_CONFIGS = {
    "drone": {"speed": 2, "capacity": 3,  "max_energy": 40},
    "robot": {"speed": 1, "capacity": 6,  "max_energy": 120},
}

LR          = 0.15
GAMMA       = 0.95
EPS_START   = 1.0
EPS_MIN     = 0.05
EPS_DECAY   = 0.992

R_DELIVER   =  15.0
R_PICKUP    =   3.0
P_STEP      =  -0.2
P_EXHAUST   = -25.0

# ── CLASSES ───────────────────────────────────────────────────────────────────

class Task:
    def __init__(self, tid, x, y):
        self.id       = tid
        self.x, self.y= x, y
        self.assigned = False
        self.done     = False

class Agent:
    def __init__(self, aid, kind, sx, sy):
        self.id   = aid
        self.kind = kind
        cfg = AGENT_CONFIGS[kind]
        self.speed      = cfg["speed"]
        self.capacity   = cfg["capacity"]
        self.max_energy = cfg["max_energy"]
        self.reset(sx, sy)

    def reset(self, sx, sy):
        self.x, self.y  = sx, sy
        self.energy     = self.max_energy
        self.cargo      = []
        self.exhausted  = False
        self.delivered  = 0

    def dist(self, x, y):
        return abs(self.x - x) + abs(self.y - y)

    def step_toward(self, tx, ty):
        taken = 0
        for _ in range(self.speed):
            if self.energy <= 0:
                self.exhausted = True
                break
            if self.x == tx and self.y == ty:
                break
            if abs(self.x - tx) >= abs(self.y - ty):
                self.x += 1 if tx > self.x else -1
            else:
                self.y += 1 if ty > self.y else -1
            self.energy -= 1
            taken += 1
        return taken

# ── ENVIRONMENT ───────────────────────────────────────────────────────────────

class Env:
    N_ACTIONS = 4   # drone-pickup, robot-pickup, drone-depot, robot-depot

    def __init__(self):
        self.drone = Agent("drone", "drone", 0, 0)
        self.robot = Agent("robot", "robot", 9, 9)
        self.tasks = []
        self.steps = 0

    def reset(self):
        self.drone.reset(0, 0)
        self.robot.reset(9, 9)
        taken = {(0,0), (9,9), DEPOT}
        positions = []
        while len(positions) < N_TASKS:
            p = (random.randint(0, GRID-1), random.randint(0, GRID-1))
            if p not in taken:
                taken.add(p)
                positions.append(p)
        self.tasks = [Task(i, x, y) for i, (x, y) in enumerate(positions)]
        self.steps = 0
        return self._state()

    def _state(self):
        remaining = sum(1 for t in self.tasks if not t.done)
        free      = [t for t in self.tasks if not t.assigned and not t.done]
        def ediscr(a): return min(4, int(a.energy / (a.max_energy / 4)))
        def ddiscr(a):
            if not free: return 0
            return min(4, min(a.dist(t.x, t.y) for t in free) // 4)
        return (
            min(4, remaining // 2),
            ediscr(self.drone), ediscr(self.robot),
            min(3, len(self.drone.cargo)), min(6, len(self.robot.cargo)),
            ddiscr(self.drone), ddiscr(self.robot),
        )

    def _pickup(self, agent, reward_acc):
        free = [t for t in self.tasks if not t.assigned and not t.done]
        if agent.exhausted:
            return reward_acc + P_STEP
        if len(agent.cargo) >= agent.capacity or not free:
            # nudge toward depot
            steps = agent.step_toward(*DEPOT)
            reward_acc += P_STEP * max(steps, 1)
            reward_acc = self._check_depot(agent, reward_acc)
            return reward_acc
        target = min(free, key=lambda t: agent.dist(t.x, t.y))
        target.assigned = True
        steps = agent.step_toward(target.x, target.y)
        reward_acc += P_STEP * max(steps, 1)
        if agent.x == target.x and agent.y == target.y:
            agent.cargo.append(target)
            reward_acc += R_PICKUP
        return reward_acc

    def _depot(self, agent, reward_acc):
        if agent.exhausted:
            return reward_acc + P_STEP
        steps = agent.step_toward(*DEPOT)
        reward_acc += P_STEP * max(steps, 1)
        return self._check_depot(agent, reward_acc)

    def _check_depot(self, agent, reward_acc):
        if agent.x == DEPOT[0] and agent.y == DEPOT[1] and agent.cargo:
            n = len(agent.cargo)
            for t in agent.cargo:
                t.done = True
            agent.delivered += n
            agent.cargo = []
            reward_acc += R_DELIVER * n
        return reward_acc

    def step(self, action):
        r = 0.0
        if   action == 0: r = self._pickup(self.drone, r)
        elif action == 1: r = self._pickup(self.robot, r)
        elif action == 2: r = self._depot(self.drone, r)
        elif action == 3: r = self._depot(self.robot, r)

        # exhaust penalty
        for ag in [self.drone, self.robot]:
            if ag.exhausted and ag.cargo:
                r += P_EXHAUST / MAX_STEPS

        self.steps += 1
        tasks_done = sum(1 for t in self.tasks if t.done)
        done = (tasks_done == N_TASKS or self.steps >= MAX_STEPS or
                (self.drone.exhausted and self.robot.exhausted))
        info = {"tasks_done": tasks_done, "steps": self.steps,
                "efficiency": tasks_done / max(self.steps, 1)}
        return self._state(), r, done, info

# ── Q-AGENT ───────────────────────────────────────────────────────────────────

class QAgent:
    def __init__(self):
        self.q   = defaultdict(lambda: np.zeros(Env.N_ACTIONS))
        self.eps = EPS_START

    def act(self, state):
        if random.random() < self.eps:
            return random.randrange(Env.N_ACTIONS)
        return int(np.argmax(self.q[state]))

    def learn(self, s, a, r, s2, done):
        target = r + (0 if done else GAMMA * np.max(self.q[s2]))
        self.q[s][a] += LR * (target - self.q[s][a])

    def decay(self):
        self.eps = max(EPS_MIN, self.eps * EPS_DECAY)

# ── BASELINES ─────────────────────────────────────────────────────────────────

def run_random(env, n=150):
    tasks = []
    for _ in range(n):
        env.reset(); done = False
        while not done:
            _, _, done, info = env.step(random.randrange(Env.N_ACTIONS))
        tasks.append(info["tasks_done"])
    return tasks

def run_greedy(env, n=150):
    tasks = []
    for _ in range(n):
        env.reset(); done = False
        while not done:
            d, rb = env.drone, env.robot
            free  = [t for t in env.tasks if not t.assigned and not t.done]
            if d.exhausted and rb.exhausted:
                action = 3
            elif len(d.cargo) >= d.capacity or (not free and d.cargo):
                action = 2
            elif len(rb.cargo) >= rb.capacity or (not free and rb.cargo):
                action = 3
            elif free and not d.exhausted and len(d.cargo) < d.capacity:
                action = 0
            elif free and not rb.exhausted and len(rb.cargo) < rb.capacity:
                action = 1
            else:
                action = 3
            _, _, done, info = env.step(action)
        tasks.append(info["tasks_done"])
    return tasks

# ── TRAINING ──────────────────────────────────────────────────────────────────

def train():
    env   = Env()
    agent = QAgent()
    ep_rewards, ep_tasks, ep_eps = [], [], []

    print(f"Training Q-agent for {N_EPS} episodes...")
    for ep in range(N_EPS):
        s = env.reset(); total = 0.0; done = False
        while not done:
            a = agent.act(s)
            s2, r, done, info = env.step(a)
            agent.learn(s, a, r, s2, done)
            s = s2; total += r
        agent.decay()
        ep_rewards.append(total)
        ep_tasks.append(info["tasks_done"])
        ep_eps.append(agent.eps)
        if (ep + 1) % 100 == 0:
            last = ep_tasks[max(0, ep-99):ep+1]
            print(f"  Ep {ep+1:4d} | avg tasks: {np.mean(last):.2f} | eps: {agent.eps:.3f} | states: {len(agent.q)}")

    print("\nRunning baselines...")
    rand_t  = run_random(env, 150)
    greed_t = run_greedy(env, 150)

    results = {"q_rewards": ep_rewards, "q_tasks": ep_tasks, "epsilon": ep_eps,
               "baseline_random_tasks": rand_t, "baseline_greedy_tasks": greed_t}
    with open("training_results.json", "w") as f:
        json.dump(results, f)
    print("Saved training_results.json")
    return results

# ── PLOTS ─────────────────────────────────────────────────────────────────────

def smooth(data, w=30):
    return np.convolve(data, np.ones(w)/w, mode="valid")

def plot(results):
    qt = results["q_tasks"]; qr = results["q_rewards"]
    rt = results["baseline_random_tasks"]; gt = results["baseline_greedy_tasks"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("KnowledgeQuarry · Problem 02 — Q-Learning Results", fontsize=14)

    ax = axes[0,0]
    ax.plot(smooth(qt), color="#378ADD", lw=2, label="Q-agent (smoothed)")
    ax.axhline(np.mean(rt),  color="#E24B4A", ls="--", label=f"Random  (mean={np.mean(rt):.1f})")
    ax.axhline(np.mean(gt),  color="#EF9F27", ls="--", label=f"Greedy  (mean={np.mean(gt):.1f})")
    ax.set_title("Tasks completed per episode (learning curve)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Tasks"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[0,1]
    ax.plot(smooth(qr), color="#1D9E75", lw=2)
    ax.set_title("Episode reward over training")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total reward"); ax.grid(alpha=0.3)

    ax = axes[1,0]
    ax.plot(results["epsilon"], color="#7F77DD", lw=1.5)
    ax.axhline(EPS_MIN, color="#aaa", ls=":")
    ax.set_title("Exploration rate (epsilon decay)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon"); ax.grid(alpha=0.3)

    ax = axes[1,1]
    methods = ["Random", "Greedy", "Q-agent\n(last 100)"]
    means   = [np.mean(rt), np.mean(gt), np.mean(qt[-100:])]
    stds    = [np.std(rt),  np.std(gt),  np.std(qt[-100:])]
    bars    = ax.bar(methods, means, yerr=stds, color=["#E24B4A","#EF9F27","#378ADD"],
                     capsize=5, width=0.45, alpha=0.85)
    for b, m in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f"{m:.1f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Method comparison — mean tasks completed")
    ax.set_ylabel("Tasks (mean ± std)"); ax.set_ylim(0, N_TASKS+1); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("kq_results.png", dpi=150, bbox_inches="tight")
    print("Saved kq_results.png")
    plt.show()

    ql = qt[-100:]
    impr = (np.mean(ql) - np.mean(rt)) / max(np.mean(rt), 0.01) * 100
    print(f"\n{'='*50}\nFINAL RESULTS\n{'='*50}")
    print(f"  Random  : {np.mean(rt):.2f} tasks/ep")
    print(f"  Greedy  : {np.mean(gt):.2f} tasks/ep")
    print(f"  Q-agent : {np.mean(ql):.2f} tasks/ep")
    print(f"  Improvement vs random: +{impr:.1f}%")

if __name__ == "__main__":
    random.seed(42); np.random.seed(42)
    results = train()
    plot(results)