"""
TestTrainedRover.py
 
Runs randomized Mars rover navigation trials using a heuristic transfer policy, 
visualizes each trial as a saved image, and optionally overlays an 
algorithmically calculated optimal path for benchmarking purposes.
 
Dependencies:
    - RoverMDP: The Markov Decision Process environment for rover navigation
    - TestRoverMars.MarsTerrain: Procedural terrain generator
    - numpy, matplotlib, heapq, os, sys
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import heapq
import sys
from RoverMDP import RoverMDP
from TestRoverMars import MarsTerrain

output_dir = "rover_final_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class RealWorldTransferPolicy:
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
        Initializes the policy with an empty visited_map.
 
        Parameters:
            height (int): Number of rows in the terrain grid.
            width (int): Number of columns in the terrain grid.
        """
        self.visited_map = np.zeros((height, width))
        self.last_goal_reached = False
        self.height = height
        self.width = width

    def get_action(self, obs, curr_y, curr_x):
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
        
        self.visited_map[curr_y, curr_x] += 1
        
        energy = obs['energy']
        objective_dist = obs['objective_distance']
        
        go_charge = False
        closest_charger = None
        
        # Determine whether to go to a charging station or not
        if not goal_reached and obs['charging_stations']: # Goal not yet reached
            closest_charger = min(obs['charging_stations'], key=lambda x: x['distance'])
            if energy < 10: # If very low, always charge
                go_charge = True
            elif energy < 35 and closest_charger['distance'] < objective_dist: # If low and a station is nearer than the objective
                go_charge = True
        elif goal_reached and obs['charging_stations']: # Goal reached
            energy_needed_estimate = objective_dist * 2.5
            if energy < energy_needed_estimate: # If has less energy that estimated to finish, go to the nearest charging station
                closest_charger = min(obs['charging_stations'], key=lambda x: x['distance'])
                go_charge = True

        target_dir = closest_charger['direction'] if go_charge else obs['objective_direction']
        
        local_grid = obs['local_grid']
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        scores = []
        
        # Create a score for each action
        for action, (dy, dx) in enumerate(deltas):
            ny, nx = curr_y + dy, curr_x + dx
            ly, lx = 2 + dy, 2 + dx
            
            # Penalize steepness
            elev = local_grid[ly, lx] if (0 <= ly < 5 and 0 <= lx < 5) else 99
            slope_penalty = 35.0 if elev > 6.0 else 0.0
            
            # Penalize visited tiles
            mem_penalty = 0
            if 0 <= ny < 20 and 0 <= nx < 30:
                mem_penalty = self.visited_map[ny, nx] * 12.0
            
            angle_diff = np.abs(target_dir - np.arctan2(dy, dx))
            
            scores.append((action, angle_diff + slope_penalty + mem_penalty))
            
        return min(scores, key=lambda x: x[1])[0]

def get_optimal_benchmark(terrain, start, goal, stations):
    """
    Computes an approximately optimal path from start to goal using an A* based search, to compare against the rover's policy.
 
    Parameters:
        terrain (np.ndarray): 2D elevation grid (height x width).
        start (tuple): Starting position as (y, x).
        goal (tuple): Goal position as (y, x).
        stations (list): List of charging station positions as (y, x) tuples.
 
    Returns:
        list: List of (y, x) positions forming the path from start to goal, or None if no valid path exists.
    """

    start_node = (0, start[0], start[1], 100, 0, [start])
    queue = [start_node]
    visited = {} 

    while queue:
        (cost, y, x, energy, mask, path) = heapq.heappop(queue)

        if (y, x) == goal: # Return goal as soon as it is found
            return path

        # Skip already visited tiles
        state_key = (y, x, mask)
        if state_key in visited and visited[state_key] <= cost:
            continue
        visited[state_key] = cost

        # Search in each direction
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < 20 and 0 <= nx < 30:
                slope = abs(terrain[ny, nx] - terrain[y, x])
                
                move_cost = 1 + (slope * 2) 
                new_energy = energy - move_cost
                
                # Only move if slope is within bounds and has energy
                if slope <= 6.0 and new_energy > 0:
                    new_mask = mask
                    final_energy = new_energy
                    
                    for i, (sy, sx) in enumerate(stations):
                        if ny == sy and nx == sx and not (mask & (1 << i)):
                            new_mask |= (1 << i)
                            final_energy = 100 
                    
                    new_cost = cost + 1
                    heapq.heappush(queue, (new_cost, ny, nx, final_energy, new_mask, path + [(ny, nx)]))
    
    return None # Default when no path is found

def get_random_spawn_config():
    """
    Returns a random (start, goal) pair from a fixed set of corner-to-corner configurations.
 
    Returns:
        tuple: A pair of coordinates for start and goal ((start_y, start_x), (goal_y, goal_x)).
    """
    configs = [((17, 2), (2, 27)), ((2, 2), (17, 27)), ((2, 2), (2, 27)), ((17, 2), (17, 27))] # Coord presets
    return configs[np.random.randint(0, len(configs))]

def export_enhanced_slideshow(n_maps=50, show_optimal=False):
    """
    Runs and visualizes n_maps randomized rover navigation trials, saving each as a PNG image in the output directory.
 
    Parameters:
        n_maps (int): Number of randomized trials to generate. Default is 50.
        show_optimal (bool): If True, compute and overlay the optimal path for comparison. Default is False.
    """
    print(f"Generating {n_maps} randomized trials. Optimal Pathing: {show_optimal}")

    for m_idx in range(n_maps):
        # Generates a seed for random terrain while keeping reproducibility
        m_seed = np.random.randint(0, 1000000)
        terrain = MarsTerrain.generate(width=30, height=20, seed=m_seed)
        start, goal = get_random_spawn_config()
        
        # Randomly places 3 charging stations on low elevation tiles
        np.random.seed(m_seed)
        stations = []
        while len(stations) < 3:
            cand = (np.random.randint(4, 16), np.random.randint(4, 26))
            if terrain[cand] < 5.0 and cand not in stations: stations.append(cand)
        
        # Initialize MDP and policy
        mdp = RoverMDP(terrain, stations, goal, start)
        policy = RealWorldTransferPolicy(20, 30)
        
        # Run episode until completion or time-out
        state = (start[0], start[1], 100, 0, 0, -1, -1, False)
        path = [start]
        for _ in range(mdp.max_time):
            obs = mdp.get_observation(state)
            action = policy.get_action(obs, state[0], state[1])
            state, _, done, _ = mdp.step(state, action)
            path.append((state[0], state[1]))
            if done: break
        
        # Generate optimal path if setting enabled
        opt_path = None
        if show_optimal:
            opt_path = get_optimal_benchmark(terrain, start, goal, stations)

        # Get how many times each tile was visited
        visit_count = {}
        for py, px in path:
            visit_count[(py, px)] = visit_count.get((py, px), 0) + 1

        # === Visualization ===
        plt.figure(figsize=(12, 7))
        plt.imshow(terrain, cmap='hot', alpha=0.7)
        
        # Overlay optimal path if setting is enabled and one exists
        if show_optimal and opt_path:
            oy, ox = zip(*opt_path)
            plt.plot(ox, oy, color='white', linestyle='--', alpha=0.8, label='Optimal Path (A*)', zorder=2)
        
        # Draw rover path
        py, px = zip(*path)
        for i in range(len(path) - 1):
            py_cur, px_cur = path[i]
            overlap = visit_count[(py_cur, px_cur)]
            if overlap == 1:
                plt.plot([px_cur], [py_cur], 'c.', markersize=6, alpha=0.7) # cyan if tile visited once
            elif overlap == 2:
                plt.plot([px_cur], [py_cur], 'y.', markersize=6, alpha=0.7) # yellow if tile visited 2 times
            else:
                plt.plot([px_cur], [py_cur], 'r.', markersize=6, alpha=0.8) # red if tile visited 3+ times
        
        plt.plot(px, py, color='cyan', linewidth=0.8, alpha=0.3)
        
        # Mark start and goal
        plt.scatter(start[1], start[0], c='green', marker='*', s=250, label='Start/Base', edgecolors='black', zorder=5)
        plt.scatter(goal[1], goal[0], c='blue', marker='*', s=250, label='Goal', edgecolors='black', zorder=5)
        
        # Mark end position if rover did not finish
        if (state[0], state[1]) != start:
            plt.scatter(state[1], state[0], c='red', marker='X', s=200, label='Failed/Died', zorder=6)
        
        # Mark charging stations
        for i, (sy, sx) in enumerate(stations):
            label = 'Charging Station' if i == 0 else ""
            plt.scatter(sx, sy, c='purple', marker='o', s=100, edgecolors='white', label=label, zorder=4)
        
        # If enabled, compute rover efficiency relative to optimal path
        eff_str = "N/A"
        if show_optimal and opt_path:
            eff_str = f"{len(opt_path)/len(path):.2%}"
            
        plt.title(f"Trial {m_idx+1} | Seed: {m_seed}\nPath Efficiency: {eff_str}")
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        plt.subplots_adjust(right=0.82) 
        
        plt.savefig(f"{output_dir}/run_{m_idx:02d}.png", bbox_inches='tight')
        plt.close()

    print(f"Done! All {n_maps} trials saved to '{output_dir}'.")

if __name__ == "__main__":
    do_optimal = False
    if len(sys.argv) > 1 and sys.argv[1].lower() == "yes":
        do_optimal = True
        
    export_enhanced_slideshow(show_optimal=do_optimal)