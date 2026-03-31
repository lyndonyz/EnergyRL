import numpy as np
import matplotlib.pyplot as plt
from MDP import MDP
from RL import RL

# === 1. Define State Space ===
# (Time: 4, Price: 2, Demand: 2, Energy: 5) = 80 total states
states = []
for time in range(4):
    for price in range(2):
        for demand in range(2):
            for energy in range(5):
                states.append((time, price, demand, energy))

state_to_index = {s: i for i, s in enumerate(states)}
nStates = len(states)
nActions = 3

# === 2. Define Transition (T) and Reward (R) Matrices ===
T = np.zeros((nActions, nStates, nStates))
R = np.zeros((nActions, nStates))

for a in range(nActions):
    for s_idx, state in enumerate(states):
        time, price, demand, energy = state
        
        # Energy Depletion Logic
        usage = a
        if energy == 0:
            usage = 0 # Cannot use what you don't have
        
        next_energy = max(0, energy - usage)
        next_time = (time + 1) % 4
        
        # Simplified environment dynamics
        next_price = 0 if next_time < 2 else 1
        
        for next_demand in [0, 1]:
            prob = 0.5 # 50/50 chance of high demand in next step
            next_state = (next_time, next_price, next_demand, next_energy)
            next_idx = state_to_index[next_state]
            T[a, s_idx, next_idx] += prob

        # Reward Logic
        reward = 0
        # High reward for meeting demand
        if demand == 1 and usage > 0:
            reward += 10
        # Penalty for missing demand
        if demand == 1 and usage == 0:
            reward -= 10
        
        # Cost of energy usage
        reward -= usage
        
        # Critical penalty for running out of energy
        if energy == 0:
            reward -= 5

        R[a, s_idx] = reward

# === 3. Solve Mathematically (Value Iteration) ===
discount = 0.9
mdp = MDP(T, R, discount)
initialV = np.zeros(nStates)
V_math, n_iters, eps = mdp.valueIteration(initialV)
policy_math = mdp.extractPolicy(V_math)

print(f"Value Iteration finished in {n_iters} iterations.")

# === 4. Solve via Learning (Q-Learning) ===
def identity_reward(meanReward):
    return meanReward

rl_agent = RL(mdp, identity_reward)
initialQ = np.zeros((nActions, nStates))

# Parameters to help the agent learn the energy constraints
nEpisodes = 10000
nSteps = 20 
epsilon = 0.05
temperature = 0.2 

Q_learned, policy_learned = rl_agent.qLearning(
    s0=0, 
    initialQ=initialQ, 
    nEpisodes=nEpisodes, 
    nSteps=nSteps, 
    epsilon=epsilon, 
    temperature=temperature
)

# === 5. Visualization ===

# Plot A: Learning Curve
plt.figure(figsize=(10, 4))
plt.plot(rl_agent.episodeRewards)
plt.title("RL Learning Curve (Energy Adaptation)")
plt.xlabel("Episode")
plt.ylabel("Cumulative Discounted Reward")
plt.grid(True, alpha=0.3)

# Plot B: Policy Comparison
plt.figure(figsize=(12, 5))
# Plotting a slice of states (where Energy varies) to see the difference
sample_range = range(0, 20) 
plt.step(sample_range, policy_math[sample_range], label="Optimal (Math)", where='mid', linewidth=2)
plt.step(sample_range, policy_learned[sample_range], label="Learned (RL)", where='mid', linestyle='--')
plt.xticks(sample_range)
plt.yticks([0, 1, 2], ['Low', 'Med', 'High'])
plt.xlabel("State Index")
plt.ylabel("Action Taken")
plt.title("Policy Comparison: Math vs. RL")
plt.legend()
plt.tight_layout()

plt.show()