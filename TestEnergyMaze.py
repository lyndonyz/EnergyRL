import numpy as np
import matplotlib.pyplot as plt
from MDP import MDP
from RL import RL

# === 1. Define Maze Layout ===
# 0 = Path (1 energy), 1 = Mud (2 energy), 2 = Trap (-50 reward), 3 = Goal (+100 reward)
maze = np.array([
    [0, 0, 0, 0, 3],
    [0, 1, 1, 0, 0],
    [0, 2, 0, 1, 0],
    [0, 0, 0, 1, 0]
])
rows, cols = maze.shape
max_energy = 10

# Create State Space: (row, col, energy)
states = []
for r in range(rows):
    for c in range(cols):
        for e in range(max_energy + 1):
            states.append((r, c, e))

state_to_idx = {s: i for i, s in enumerate(states)}
nStates = len(states)
nActions = 4 # 0: Up, 1: Down, 2: Left, 3: Right

# === 2. Define MDP Dynamics ===
T = np.zeros((nActions, nStates, nStates))
R = np.zeros((nActions, nStates))

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # row/col offsets

for s_idx, (r, c, e) in enumerate(states):
    # If already at Goal or Trap, state is terminal (self-loop)
    if maze[r, c] == 2 or maze[r, c] == 3 or e == 0:
        for a in range(nActions):
            T[a, s_idx, s_idx] = 1.0
        continue

    for a, (dr, dc) in enumerate(actions):
        nr, nc = r + dr, c + dc
        
        # Boundary Check
        if 0 <= nr < rows and 0 <= nc < cols:
            # Determine Energy Cost based on tile type
            tile_type = maze[nr, nc]
            cost = 2 if tile_type == 1 else 1
            ne = max(0, e - cost)
            
            next_idx = state_to_idx[(nr, nc, ne)]
            T[a, s_idx, next_idx] = 1.0
            
            # Reward Logic
            if tile_type == 3: # Goal
                R[a, s_idx] = 100
            elif tile_type == 2: # Trap
                R[a, s_idx] = -50
            elif ne == 0 and tile_type != 3: # Out of energy
                R[a, s_idx] = -20
            else:
                R[a, s_idx] = -1 # Small step penalty to encourage speed
        else:
            # Hit a wall: stay in place, lose 1 energy
            ne = max(0, e - 1)
            next_idx = state_to_idx[(r, c, ne)]
            T[a, s_idx, next_idx] = 1.0
            R[a, s_idx] = -2

# === 3. Run RL Agent ===
def identity_reward(r): return r
mdp = MDP(T, R, discount=0.95)
rl_agent = RL(mdp, identity_reward)

# Start at (0,0) with full energy
s0 = state_to_idx[(0, 0, max_energy)]
Q, policy = rl_agent.qLearning(s0, np.zeros((nActions, nStates)), nEpisodes=5000, nSteps=50, epsilon=0.1)

# === 4. Visualize the Path ===
def plot_maze_path(maze, policy, states):
    path_mask = np.zeros_like(maze, dtype=float)
    curr_s = (0, 0, max_energy)
    
    for _ in range(20):
        r, c, e = curr_s
        path_mask[r, c] += 1
        if maze[r, c] in [2, 3] or e == 0: break
        
        a = policy[state_to_idx[curr_s]]
        dr, dc = [(-1,0), (1,0), (0,-1), (0,1)][a]
        # Simplified move for visualization
        nr, nc = np.clip(r+dr, 0, rows-1), np.clip(c+dc, 0, cols-1)
        curr_s = (nr, nc, max(0, e-1))

    plt.imshow(maze, cmap='cool')
    plt.title("Maze: Goal(3), Trap(2), Mud(1)")
    plt.show()

plot_maze_path(maze, policy, states)

def plot_learning_results(rl_agent):
    # 1. Extract raw rewards from the agent
    rewards = rl_agent.episodeRewards
    
    # 2. Calculate a Moving Average (window of 50 episodes)
    window_size = 50
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    else:
        moving_avg = rewards # Fallback if training was too short

    # 3. Create the Plot
    plt.figure(figsize=(10, 6))
    
    # Plot raw data with low opacity (the "noise")
    plt.plot(rewards, color='blue', alpha=0.2, label='Raw Episode Reward')
    
    # Plot the smoothed trend line (the "learning")
    plt.plot(range(window_size-1, len(rewards)), moving_avg, color='red', linewidth=2, label=f'Moving Average ({window_size})')
    
    plt.title("RL Agent Learning Curve: Maze Navigation")
    plt.xlabel("Episode Number")
    plt.ylabel("Cumulative Discounted Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Call the function after your training is finished
plot_learning_results(rl_agent)