"""
TestRoverMars.py
 
Main training and visualization script for the Mars Rover Q-learning agent.
 
Usage:
    python TestRoverMars.py          # Run without saving images
    python TestRoverMars.py yes      # Run and save milestone images
 
Dependencies:
    - RoverMDP: The Markov Decision Process environment for rover navigation
    - RL: Reinforcement learning helper (used for agent setup)
    - numpy, matplotlib
"""

import numpy as np
from RoverMDP import RoverMDP
from RL import RL
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

class RealWorldRoverPolicy:
    """
    Heuristic navigation policy designed to transfer to conditions similar to the real world.

    Attributes:
        visited_map (np.ndarray): Tracks how many times each cell has been visited.
        last_goal_reached (bool): If previous step reached goal, used to reset visited tiles for return leg.
        height (int): Grid height.
        width (int): Grid width.
    """

    def __init__(self, height=20, width=30):
        """
        Initializes the policy with an empty visited_map 

        Parameters:
            height (int): Number of rows in the terrain grid.
            width (int): Number of columns in the terrain grid.
        """
        self.visited_map = np.zeros((height, width))
        self.last_goal_reached = False
        self.height = height
        self.width = width

    def get_action(self, obs, current_y, current_x):
        """
        Selects the best action for the current step using a scoring heuristic.
 
        Parameters:
            obs (dict): Observation dictionary from RoverMDP.get_observation().
            curr_y (int): Current y position of the rover.
            curr_x (int): Current x position of the rover.
 
        Returns:
            int: The action index with the lowest score.
        """

        goal_reached = obs['goal_reached']
        
        # Reset visited map on goal reach, so the rover can return to start properly
        if goal_reached != self.last_goal_reached:
            self.visited_map = np.zeros((self.height, self.width))
            self.last_goal_reached = goal_reached
        
        self.visited_map[current_y, current_x] += 1
        
        energy = obs['energy']
        objective_dist = obs['objective_distance']
        
        go_charge = False
        closest_charger = None
        
        # Determine whether to go to a charging station or not
        if not goal_reached and obs['charging_stations']:# Goal not yet reached
            closest_charger = min(obs['charging_stations'], key=lambda x: x['distance'])
            if energy < 10: # If very low, always charge
                go_charge = True
            elif energy < 35 and closest_charger['distance'] < objective_dist: # If low and a station is nearer than the objective
                go_charge = True
        elif goal_reached and obs['charging_stations']: # Goal reached
            energy_needed_estimate = objective_dist * 2.5
            if energy < energy_needed_estimate:# If has less energy that estimated to finish, go to the nearest charging station
                closest_charger = min(obs['charging_stations'], key=lambda x: x['distance'])
                go_charge = True
        
        if go_charge:
            target_dir = closest_charger['direction']
        else:
            target_dir = obs['objective_direction']
        
        local_grid = obs['local_grid']
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        scores = []
        
        # Create a score for each action
        for action, (dy, dx) in enumerate(deltas):
            ny, nx = current_y + dy, current_x + dx
            ly, lx = 2 + dy, 2 + dx
            
            # Penalize steepness
            elev = local_grid[ly, lx] if (0 <= ly < 5 and 0 <= lx < 5) else 99
            slope_penalty = 35.0 if elev > 6.0 else 0.0
            angle_diff = np.abs(target_dir - np.arctan2(dy, dx))
            
            # Penalize visited tiles
            if 0 <= ny < self.visited_map.shape[0] and 0 <= nx < self.visited_map.shape[1]:
                memory_penalty = self.visited_map[ny, nx] * 12.0
            else:
                memory_penalty = 100
            
            scores.append((action, angle_diff + slope_penalty + memory_penalty))
            
        return min(scores, key=lambda x: x[1])[0]

def run_mission(terrain, stations, goal, start):
    """
    Runs a single mission episode using the heuristic RealWorldRoverPolicy.
 
    Parameters:
        terrain (np.ndarray): 2D elevation grid.
        stations (list): List of charging station positions.
        goal (tuple): Goal position.
        start (tuple): Starting position.
 
    Returns:
        tuple:
            - path (list of (y, x)): All positions visited during the episode.
            - state (tuple): The final MDP state at episode end.
    """

    mdp = RoverMDP(terrain, stations, goal, start)
    policy = RealWorldRoverPolicy(height=terrain.shape[0], width=terrain.shape[1])
    
    # Initialize state and path
    state = (start[0], start[1], 100, 0, 0, -1, -1, False)
    path = [start]
    
    # Run episode until completion or time-out
    for _ in range(mdp.max_time):
        obs = mdp.get_observation(state)
        curr_y, curr_x = state[0], state[1]
        action = policy.get_action(obs, curr_y, curr_x)
        next_state, reward, done, _ = mdp.step(state, action)
        state = next_state
        path.append((state[0], state[1]))
        if done: break
        
    return path, state

class MarsTerrain:
    """Procedural Mars terrain generator."""

    @staticmethod
    def generate(width=30, height=20, seed=42):
        """
        Generates a randomized terrain grid.
 
        Parameters:
            width (int): Number of columns in the grid. Default is 30.
            height (int): Number of rows in the grid. Default is 20.
            seed (int): Random seed for reproducibility. Default is 42.
 
        Returns:
            np.ndarray: A (height x width) float32 elevation array with values in [0, 10].
        """
        np.random.seed(seed)
        terrain = np.zeros((height, width))
        terrain[:] = 2.5 # Default elevation
        
        # Add ridges
        for _ in range(5):
            ridge_y = np.random.randint(2, height - 2)
            ridge_x = np.random.randint(2, width - 2)
            ridge_radius = np.random.randint(4, 8)  
            peak_height = np.random.uniform(6.0, 8.0)
            for y in range(height):
                for x in range(width):
                    dist = np.sqrt((y - ridge_y)**2 + (x - ridge_x)**2)
                    if dist < ridge_radius:
                        elevation = peak_height * np.exp(-(dist**2) / (ridge_radius**2 * 0.5))
                        terrain[y, x] = max(terrain[y, x], elevation)
        
        # Add Craters
        for _ in range(6):
            crater_y = np.random.randint(2, height - 2)
            crater_x = np.random.randint(2, width - 2)
            crater_radius = np.random.randint(2, 5)
            crater_depth = np.random.uniform(0.5, 2.0)
            for y in range(height):
                for x in range(width):
                    dist = np.sqrt((y - crater_y)**2 + (x - crater_x)**2)
                    if dist < crater_radius:
                        elevation = crater_depth * np.exp(-(dist**2) / (crater_radius**2 * 0.5))
                        terrain[y, x] = min(terrain[y, x], 2.5 - elevation)
        
        # Add random noise
        for y in range(height):
            for x in range(width):
                terrain[y, x] += np.random.normal(0, 0.15)
        
        return np.clip(terrain, 0, 10).astype(np.float32)

def create_rover_mdp(seed=42):
    """
    Creates and returns a configured RoverMDP instance with a fixed terrain and layout.
 
    Parameters:
        seed (int): Random seed for terrain generation. Default is 42.
 
    Returns:
        tuple:
            - mdp (RoverMDP): The configured MDP environment.
            - terrain (np.ndarray): The generated terrain grid.
            - start (tuple): Starting position (y, x).
            - goal (tuple): Goal position (y, x).
            - charging_stations (list): List of charging station positions.
    """
    terrain = MarsTerrain.generate(width=30, height=20, seed=seed)
    start = (19, 1) # Bottom left
    goal = (0, 28) # Top right
    charging_stations = [(10, 8), (5, 20), (15, 15)]
    mdp = RoverMDP(terrain=terrain, charging_stations=charging_stations, goal=goal, start=start, max_energy=100, max_time=600, max_slope=2.0)
    return mdp, terrain, start, goal, charging_stations

def save_milestone_image(episode, path, terrain, start, goal, stations, reward, folder="training_milestones"):
    """
    Saves a PNG image of the rover's path at a training milestone.
 
    Parameters:
        episode (int): Current episode number (used in the filename and title).
        path (list of (y, x)): Sequence of positions visited during the episode.
        terrain (np.ndarray): 2D elevation grid.
        start (tuple): Start position (y, x).
        goal (tuple): Goal position (y, x).
        stations (list): Charging station positions as (y, x) tuples.
        reward (float): Total reward earned this episode (shown in title).
        folder (str): Output directory for saved images. Default is "training_milestones".
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.figure(figsize=(10, 7))
    plt.imshow(terrain, cmap='hot', origin='upper', alpha=0.7)
    py, px = zip(*path)
    plt.plot(px, py, 'c-', linewidth=2, label='Episode Path')
    plt.plot(start[1], start[0], 'g*', markersize=15, label='Start')
    plt.plot(goal[1], goal[0], 'b*', markersize=15, label='Goal')
    for cs in stations:
        plt.plot(cs[1], cs[0], 'o', color='purple', markersize=10)
    plt.title(f"Episode {episode} | Reward: {reward:.2f}")
    plt.savefig(f"{folder}/episode_{episode:04d}.png")
    plt.close()

def visualize_terrain(terrain, start, goal, charging_stations):
    """
    Displays a static heatmap of the terrain with start, goal, and charging stations marked.
 
    Parameters:
        terrain (np.ndarray): 2D elevation grid.
        start (tuple): Start position (y, x).
        goal (tuple): Goal position (y, x).
        charging_stations (list): Charging station positions as (y, x) tuples.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(terrain, cmap='hot', origin='upper')
    plt.colorbar(im, ax=ax, label='Elevation (0-10)')
    ax.plot(start[1], start[0], 'g*', markersize=20, label='Start')
    ax.plot(goal[1], goal[0], 'b*', markersize=20, label='Goal')
    for i, cs in enumerate(charging_stations):
        ax.plot(cs[1], cs[0], 'o', color='purple', markersize=12, label='Charging Station' if i == 0 else '')
    ax.set_xlabel('X Position (meters)')
    ax.set_ylabel('Y Position (meters)')
    ax.set_title('Mars Rover Terrain Map (30m x 20m)')
    ax.legend(loc='upper left', bbox_to_anchor=(-0.15, 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_rover_path_with_overlaps(terrain, start, goal, charging_stations, states_visited):
    """
    Visualizes the rover's path with color-coded overlap detection.
 
    Parameters:
        terrain (np.ndarray): 2D elevation grid.
        start (tuple): Start position (y, x).
        goal (tuple): Goal position (y, x).
        charging_stations (list): Charging station positions as (y, x) tuples.
        states_visited (list of (y, x)): All positions visited during the episode.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(terrain, cmap='hot', origin='upper', alpha=0.7)
    
    if len(states_visited) > 0:
        path_y = [s[0] for s in states_visited]
        path_x = [s[1] for s in states_visited]
        
        # Count number of times each tile was visited
        visit_count = {}
        for i, (py, px) in enumerate(states_visited):
            visit_count[(py, px)] = visit_count.get((py, px), 0) + 1
        
        # Draw rover path
        for i in range(len(states_visited) - 1):
            py, px = states_visited[i]
            overlap = visit_count[(py, px)]
            if overlap == 1:
                ax.plot([px], [py], 'c.', markersize=6, alpha=0.7) # cyan if tile visited once
            elif overlap == 2:
                ax.plot([px], [py], 'y.', markersize=6, alpha=0.7) # yellow if tile visited 2 times
            else:
                ax.plot([px], [py], 'r.', markersize=6, alpha=0.8) # red if tile visited 3+ times
        
        ax.plot(path_x, path_y, 'c-', linewidth=1.5, alpha=0.4, label='Path')
    
    ax.plot(start[1], start[0], 'g*', markersize=20, label='Start/Base', zorder=5)
    ax.plot(goal[1], goal[0], 'b*', markersize=20, label='Goal', zorder=5)
    for cs in charging_stations:
        ax.plot(cs[1], cs[0], 'o', color='purple', markersize=12, zorder=4)
    
    ax.set_xlabel('X Position (meters)')
    ax.set_ylabel('Y Position (meters)')
    ax.set_title('Rover Path Visualization (with Overlap Detection)')
    cyan_patch = mpatches.Patch(color='cyan', label='Single visit')
    yellow_patch = mpatches.Patch(color='yellow', label='Double visit')
    red_patch = mpatches.Patch(color='red', label='Triple+ visits')
    ax.legend(handles=[cyan_patch, yellow_patch, red_patch] + [ax.get_legend_handles_labels()[0][i] for i in range(3)], 
              loc='upper left', bbox_to_anchor=(-0.15, 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_learning_curve(episode_rewards, window_size=50):
    """
    Plots total reward per episode alongside a smoothed moving average.
 
    Parameters:
        episode_rewards (list of float): Total reward collected each episode.
        window_size (int): Number of episodes to average over. Default is 50.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, color='blue', alpha=0.3, linewidth=1, label='Episode Reward')
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')
    plt.xlabel('Episode Number')
    plt.ylabel('Total Reward')
    plt.title('Mars Rover Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_energy_efficiency(episode_energy_used, window_size=50):
    """
    Plots total energy consumed per episode alongside a smoothed moving average.
 
    Parameters:
        episode_energy_used (list of float): Total energy consumed each episode.
        window_size (int): Number of episodes to average over. Default is 50.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(episode_energy_used, color='orange', alpha=0.3, linewidth=1, label='Energy Used per Episode')
    if len(episode_energy_used) >= window_size:
        moving_avg = np.convolve(episode_energy_used, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_energy_used)), moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')
    plt.xlabel('Episode Number')
    plt.ylabel('Total Energy Used')
    plt.title('Mars Rover Energy Efficiency (Lower is Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Check if we should save images
    save_enabled = len(sys.argv) > 1 and sys.argv[1].lower() == "yes"
    
    # Set up terrain, MDP, and RL agent
    mdp, terrain, start, goal, charging_stations = create_rover_mdp(seed=42)
    visualize_terrain(terrain, start, goal, charging_stations)
    rl_agent = RL(mdp, lambda r: r)
    initial_state = (start[0], start[1], mdp.max_energy, 0, 0, -1, -1, False)
    
    # Training configuration
    n_episodes = 8000
    episode_rewards, episode_energy_used = [], []
    best_reward, best_path = -np.inf, None
    Q_table, learning_rate, discount_factor = {}, 0.1, 0.9
    
    def discretize_state(state):
        """
        Reduces the continuous MDP state to a compact discrete key for the Q-table.
 
        Parameters:
            state (tuple): Full MDP state tuple.
 
        Returns:
            tuple: (y, x, energy_bucket, visited_mask, last_action, goal_reached)
        """
        y, x, energy, time_step, visited_mask, last_action, last_dir, goal_reached = state
        return (int(y), int(x), int(energy / 20), visited_mask, last_action, goal_reached)
    
    
    def plot_efficiency_vs_reward_overlay(episode_rewards, episode_energy_used, window_size=100):
        """
        Plots energy efficiency and reward on a dual-axis chart for direct comparison.
 
        Parameters:
            episode_rewards (list of float): Total reward per episode.
            episode_energy_used (list of float): Total energy used per episode.
            window_size (int): Smoothing window size. Default is 100.
        """
        if len(episode_rewards) < window_size: return # If not enough data to plot meaningful averages, exit

        avg_reward = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        avg_energy = np.convolve(episode_energy_used, np.ones(window_size)/window_size, mode='valid')
        x_axis = range(window_size - 1, len(episode_rewards))
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Left axis: energy used
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Avg Energy Used', color='tab:orange', fontsize=12, fontweight='bold')
        line1 = ax1.plot(x_axis, avg_energy, color='tab:orange', linewidth=3, label='Avg Energy Used')
        ax1.tick_params(axis='y', labelcolor='tab:orange')
        ax1.grid(True, alpha=0.3)

        # Right axis: reward (shares x-axis with energy)
        ax2 = ax1.twinx() 
        ax2.set_ylabel('Avg Reward', color='tab:blue', fontsize=12, fontweight='bold')
        line2 = ax2.plot(x_axis, avg_reward, color='tab:blue', linewidth=3, label='Avg Reward')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        lines = line1 + line2
        ax1.legend(lines, [l.get_label() for l in lines], loc='center right')
        plt.title(f'Optimization Summary: Energy vs. Reward (Window={window_size})')
        fig.tight_layout()
        plt.show()

    # === Q-learning Training Loop ===
    for episode in range(n_episodes):
        state = initial_state
        episode_reward = episode_total_energy_used = steps_in_episode = 0
        visited_positions = [(state[0], state[1])]

        # Epsilon decay
        epsilon = max(0.01, 0.25 * (0.9992 ** episode))
        
        while steps_in_episode < mdp.max_time:
            obs = mdp.get_observation(state)
            discrete_state = discretize_state(state)

            # Initialize Q-values for unseen states
            if discrete_state not in Q_table: Q_table[discrete_state] = {a: 0.0 for a in range(4)}
            
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                # Random
                action = np.random.randint(0, 4)
            else:
                # Greedy
                best_actions = [a for a in range(4) if Q_table[discrete_state][a] == max(Q_table[discrete_state].values())]
                if len(best_actions) > 1:
                    objective_direction = obs['objective_direction']
                    action_directions = [-np.pi/2, np.pi/2, np.pi, 0]
                    direction_alignment = {a: -abs(np.arctan2(np.sin(objective_direction - action_directions[a]), np.cos(objective_direction - action_directions[a]))) for a in best_actions}
                    action = max(best_actions, key=lambda a: direction_alignment[a])
                else:
                    action = best_actions[0]
            
            # If the chosen action is invalid, fall back to the best valid action by direction
            y, x, last_action = state[0], state[1], state[5]
            is_valid, _, _ = mdp.calculate_energy_cost(y, x, action, last_action)
            if not is_valid:
                objective_direction, action_directions = obs['objective_direction'], [-np.pi/2, np.pi/2, np.pi, 0]
                valid_alts = [(a, -abs(np.arctan2(np.sin(objective_direction - action_directions[a]), np.cos(objective_direction - action_directions[a])))) for a in range(4) if mdp.calculate_energy_cost(y, x, a, last_action)[0]]
                action = max(valid_alts, key=lambda x: x[1])[0] if valid_alts else np.random.randint(0, 4)
            
            # Take the step and collect transition data
            next_state, reward, done, info = mdp.step(state, action)
            episode_reward += reward
            episode_total_energy_used += info.get('energy_used', 0)
            steps_in_episode += 1
            visited_positions.append((next_state[0], next_state[1]))
            
            # Q-learning update based on equation: Q(s,a) ← Q(s,a) + α * (r + γ * max Q(s') - Q(s,a))
            next_ds = discretize_state(next_state)
            if next_ds not in Q_table: Q_table[next_ds] = {a: 0.0 for a in range(4)}
            Q_table[discrete_state][action] += max(0.01, learning_rate * (0.9992 ** episode)) * (reward + discount_factor * max(Q_table[next_ds].values()) - Q_table[discrete_state][action])
            
            state = next_state
            if done: break
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_energy_used.append(episode_total_energy_used)
        
        # Save image every 100 episodes if enabled
        if save_enabled and (episode + 1) % 100 == 0:
            save_milestone_image(episode + 1, visited_positions, terrain, start, goal, charging_stations, episode_reward)
            
        if episode_reward > best_reward:
            best_reward, best_path = episode_reward, visited_positions
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{n_episodes} - Avg Reward: {np.mean(episode_rewards[-100:]):.2f} | Best: {best_reward:.2f}")
    
    # === Visualization ===
    visualize_rover_path_with_overlaps(terrain, start, goal, charging_stations, best_path)
    plot_learning_curve(episode_rewards, window_size=50)
    plot_energy_efficiency(episode_energy_used, window_size=50)
    plot_efficiency_vs_reward_overlay(episode_rewards, episode_energy_used, window_size=50)