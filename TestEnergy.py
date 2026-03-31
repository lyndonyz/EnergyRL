import numpy as np
from MDP import MDP

states = []
for time in range(4):
    for price in range(2):
        for demand in range(2):
            for energy in range(5):
                states.append((time, price, demand, energy))

state_to_index = {s: i for i, s in enumerate(states)}

nStates = len(states)
nActions = 3  # 0=low usage, 1=medium, 2=high

T = np.zeros((nActions, nStates, nStates))
R = np.zeros((nActions, nStates))

for a in range(nActions):
    for s_idx, state in enumerate(states):
        time, price, demand, energy = state
        usage = a

        if energy == 0:
            usage = 0

        next_energy = max(0, energy - usage)
        next_time = (time + 1) % 4
        next_price = 0 if next_time < 2 else 1

        for next_demand in [0, 1]:
            prob = 0.5
            next_state = (next_time, next_price, next_demand, next_energy)
            next_idx = state_to_index[next_state]
            T[a, s_idx, next_idx] += prob

        reward = 0

        if demand == 1 and usage > 0:
            reward += 10

        if demand == 1 and usage == 0:
            reward -= 10

        reward -= 2 * usage

        if energy == 0:
            reward -= 5

        R[a, s_idx] = reward

discount = 0.9
mdp = MDP(T, R, discount)
initialV = np.zeros(nStates)
V, _, _ = mdp.valueIteration(initialV)
policy = mdp.extractPolicy(V)

print("\nSample of Optimal Policy:\n")

for i in range(10):
    print(f"State {states[i]} -> Action {policy[i]}")