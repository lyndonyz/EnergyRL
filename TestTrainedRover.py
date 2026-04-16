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
    def __init__(self, height=20, width=30):
        self.visited_map = np.zeros((height, width))
        self.last_goal_reached = False
        self.height = height
        self.width = width

    def get_action(self, obs, curr_y, curr_x):
        goal_reached = obs['goal_reached']
        
        if goal_reached != self.last_goal_reached:
            self.visited_map = np.zeros((self.height, self.width))
            self.last_goal_reached = goal_reached
        
        self.visited_map[curr_y, curr_x] += 1
        
        energy = obs['energy']
        objective_dist = obs['objective_distance']
        
        go_charge = False
        closest_charger = None
        
        if not goal_reached and obs['charging_stations']:
            closest_charger = min(obs['charging_stations'], key=lambda x: x['distance'])
            if energy < 10:
                go_charge = True
            elif energy < 35 and closest_charger['distance'] < objective_dist:
                go_charge = True
        elif goal_reached and obs['charging_stations']:
            energy_needed_estimate = objective_dist * 2.5
            if energy < energy_needed_estimate:
                closest_charger = min(obs['charging_stations'], key=lambda x: x['distance'])
                go_charge = True

        target_dir = closest_charger['direction'] if go_charge else obs['objective_direction']
        
        local_grid = obs['local_grid']
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        scores = []
        
        for action, (dy, dx) in enumerate(deltas):
            ny, nx = curr_y + dy, curr_x + dx
            ly, lx = 2 + dy, 2 + dx
            
            elev = local_grid[ly, lx] if (0 <= ly < 5 and 0 <= lx < 5) else 99
            slope_penalty = 35.0 if elev > 6.0 else 0.0
            
            mem_penalty = 0
            if 0 <= ny < 20 and 0 <= nx < 30:
                mem_penalty = self.visited_map[ny, nx] * 12.0
            
            angle_diff = np.abs(target_dir - np.arctan2(dy, dx))
            
            scores.append((action, angle_diff + slope_penalty + mem_penalty))
            
        return min(scores, key=lambda x: x[1])[0]

def get_optimal_benchmark(terrain, start, goal, stations):
    start_node = (0, start[0], start[1], 100, 0, [start])
    queue = [start_node]
    visited = {} 

    while queue:
        (cost, y, x, energy, mask, path) = heapq.heappop(queue)

        if (y, x) == goal:
            return path

        state_key = (y, x, mask)
        if state_key in visited and visited[state_key] <= cost:
            continue
        visited[state_key] = cost

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < 20 and 0 <= nx < 30:
                slope = abs(terrain[ny, nx] - terrain[y, x])
                
                move_cost = 1 + (slope * 2) 
                new_energy = energy - move_cost
                
                if slope <= 6.0 and new_energy > 0:
                    new_mask = mask
                    final_energy = new_energy
                    
                    for i, (sy, sx) in enumerate(stations):
                        if ny == sy and nx == sx and not (mask & (1 << i)):
                            new_mask |= (1 << i)
                            final_energy = 100 
                    
                    new_cost = cost + 1
                    heapq.heappush(queue, (new_cost, ny, nx, final_energy, new_mask, path + [(ny, nx)]))
    return None

def get_random_spawn_config():
    configs = [((17, 2), (2, 27)), ((2, 2), (17, 27)), ((2, 2), (2, 27)), ((17, 2), (17, 27))]
    return configs[np.random.randint(0, len(configs))]

def export_enhanced_slideshow(n_maps=50, show_optimal=False):
    print(f"Generating {n_maps} randomized trials. Optimal Pathing: {show_optimal}")

    for m_idx in range(n_maps):
        m_seed = np.random.randint(0, 1000000)
        terrain = MarsTerrain.generate(width=30, height=20, seed=m_seed)
        start, goal = get_random_spawn_config()
        
        np.random.seed(m_seed)
        stations = []
        while len(stations) < 3:
            cand = (np.random.randint(4, 16), np.random.randint(4, 26))
            if terrain[cand] < 5.0 and cand not in stations: stations.append(cand)
        
        mdp = RoverMDP(terrain, stations, goal, start)
        policy = RealWorldTransferPolicy(20, 30)
        
        state = (start[0], start[1], 100, 0, 0, -1, -1, False)
        path = [start]
        for _ in range(mdp.max_time):
            obs = mdp.get_observation(state)
            action = policy.get_action(obs, state[0], state[1])
            state, _, done, _ = mdp.step(state, action)
            path.append((state[0], state[1]))
            if done: break
            
        opt_path = None
        if show_optimal:
            opt_path = get_optimal_benchmark(terrain, start, goal, stations)

        visit_count = {}
        for py, px in path:
            visit_count[(py, px)] = visit_count.get((py, px), 0) + 1

        plt.figure(figsize=(12, 7))
        plt.imshow(terrain, cmap='hot', alpha=0.7)
        
        if show_optimal and opt_path:
            oy, ox = zip(*opt_path)
            plt.plot(ox, oy, color='white', linestyle='--', alpha=0.8, label='Optimal Path (A*)', zorder=2)
        
        py, px = zip(*path)
        for i in range(len(path) - 1):
            py_cur, px_cur = path[i]
            overlap = visit_count[(py_cur, px_cur)]
            if overlap == 1:
                plt.plot([px_cur], [py_cur], 'c.', markersize=6, alpha=0.7)
            elif overlap == 2:
                plt.plot([px_cur], [py_cur], 'y.', markersize=6, alpha=0.7)
            else:
                plt.plot([px_cur], [py_cur], 'r.', markersize=6, alpha=0.8)
        
        plt.plot(px, py, color='cyan', linewidth=0.8, alpha=0.3)
        
        plt.scatter(start[1], start[0], c='green', marker='*', s=250, label='Start/Base', edgecolors='black', zorder=5)
        plt.scatter(goal[1], goal[0], c='blue', marker='*', s=250, label='Goal', edgecolors='black', zorder=5)
        
        if (state[0], state[1]) != start:
            plt.scatter(state[1], state[0], c='red', marker='X', s=200, label='Failed/Died', zorder=6)
        
        for i, (sy, sx) in enumerate(stations):
            label = 'Charging Station' if i == 0 else ""
            plt.scatter(sx, sy, c='purple', marker='o', s=100, edgecolors='white', label=label, zorder=4)
        
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