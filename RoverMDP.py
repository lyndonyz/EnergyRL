import numpy as np
from collections import namedtuple

class RoverMDP:
    """Represents a rover made to move through a grid simulation."""

    ACTION_NAMES = ["Up", "Down", "Left", "Right"]
    DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def __init__(self, terrain, charging_stations, goal, start, max_energy=100, max_time=500, max_slope=2.0):
        """
        Constructor for the RoverMDP class.

        Parameters:
            terrain (MarsTerrain): A 2d simulation of terrain with elevation.
            charging_stations (list): A list of the locations of all charging stations.
            goal (tuple): The rover's objective's location.
            start (tuple): The rover's starting location.
            max_energy (int): The rover's maximum energy.
            max_time (int): The maximum time the rover is allowed to move.
            max_slope (float): The steepest slop the rover can traverse.
        """

        self.height, self.width = terrain.shape
        assert self.width == 30 and self.height == 20, "Map must be 30x20"
        self.max_slope = max_slope
        
        self.terrain = terrain
        self.charging_stations = charging_stations
        self.n_charging_stations = len(charging_stations)
        self.charging_station_dict = {cs: i for i, cs in enumerate(charging_stations)}
        
        self.goal = goal
        self.start = start
        self.max_energy = max_energy
        self.max_time = max_time
        
        self.energy_move = 1.0  
        self.energy_turn = 1.2  
        self.energy_stop = 0.3  
        self.energy_elevation_factor = 0.15  
        self.energy_per_step = 0.1  
    
    def get_elevation_cost(self, from_y, from_x, to_y, to_x):
        """
        Gets the cost of moving the rover from one elevation to another.

        Parameters:
            from_y (int): y coordinate of starting position.
            from_x (int): x coordinate of starting position.
            to_y (int): y coordinate of ending position.
            to_x (int): x coordinate of ending position.
        """

        if not (0 <= to_y < self.height and 0 <= to_x < self.width):
            return 0
        
        elev_from = self.terrain[from_y, from_x]
        elev_to = self.terrain[to_y, to_x]
        
        elev_diff = max(0, elev_to - elev_from)
        return elev_diff * self.energy_elevation_factor
    
    def can_charge_at(self, y, x, visited_mask):
        """
        Checks if the rover can charge at the specified coordinate position.

        Parameters:
            y (int): y coordinate of search position.
            x (int): x coordinate of search position.
            visited_mask: mask of positions rover has already been to.
        
        Returns:
            boolean: true if the rover can charge at (x,y), false otherwise.
        """

        if (y, x) == self.start: # Automatically true on starting space
            return True
        
        if (y, x) in self.charging_station_dict:
            station_idx = self.charging_station_dict[(y, x)]
            return (visited_mask & (1 << station_idx)) == 0
        
        return False
    
    def calculate_energy_cost(self, curr_y, curr_x, action, last_action):
        """
        Calculates the energy cost of the rover taking a move.

        Parameters:
            curr_y: The rover's current y coordinate.
            curr_x: The rover's current x coordinate.
            action: The current action to be taken.
            last_action: The previous action taken.

        Returns:
            tuple: (boolean - if the move is valid, tuple - movement delta, float - move energy cost)
        """
        dy, dx = self.DELTAS[action]
        next_y, next_x = curr_y + dy, curr_x + dx
        
        # Invalid move - out of bounds
        if not (0 <= next_y < self.height and 0 <= next_x < self.width):
            return False, (0, 0), self.energy_stop + self.energy_per_step
        
        elev_from = self.terrain[curr_y, curr_x]
        elev_to = self.terrain[next_y, next_x]
        elev_diff = elev_to - elev_from
        
        # Invalid move - too steep
        if elev_diff > self.max_slope:
            return False, (0, 0), self.energy_stop + self.energy_per_step
        
        energy_cost = self.energy_move
        
        # Uphill move
        if elev_diff > 0:  
            slope_angle = np.degrees(np.arctan(elev_diff / 1.0))
            
            if slope_angle <= 5:
                multiplier = 1.0 + (slope_angle / 5.0) * 0.2
            elif slope_angle <= 10:
                multiplier = 1.2 + ((slope_angle - 5) / 5.0) * 0.3
            elif slope_angle <= 15:
                multiplier = 1.5 + ((slope_angle - 10) / 5.0) * 1.0
            else:
                multiplier = 2.5 + ((slope_angle - 15) / 10.0) * 1.5
            
            energy_cost += elev_diff * self.energy_elevation_factor * multiplier

        # Downhill move  
        elif elev_diff < 0:  
            abs_elev_diff = -elev_diff
            slope_angle = np.degrees(np.arctan(abs_elev_diff / 1.0))
            
            if abs_elev_diff <= 0.3:
                downhill_mult = 0.9 - (abs_elev_diff / 0.3) * 0.2
            elif abs_elev_diff <= 1.0:
                downhill_mult = 0.8 - ((abs_elev_diff - 0.3) / 0.7) * 0.3
            else:
                downhill_mult = 0.6 - ((abs_elev_diff - 1.0) / 1.0) * 0.3
                downhill_mult = max(0.3, downhill_mult)  
            
            energy_cost *= downhill_mult
        
        # Energy penalty for turning
        if last_action != -1 and last_action != action:
            opposite_pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
            if (last_action, action) in opposite_pairs:
                energy_cost += 1.8  
            else:
                energy_cost += 1.4  
        
        # Valid move
        return True, (dy, dx), energy_cost
    
    def step(self, state, action):
        """
        Advances the simulation by one step.

        Parameters:
            state (tuple): The rover's current state.
            action (int): The rovers intended move.
        
        Returns:
            tuple: (tuple - rover's next state, float - reward this step, boolean - if the episode ends for any reason, dict - containing diagnostic informaion)
        """
        y, x, energy, time_step, visited_mask, last_action, last_dir, goal_reached = state
        
        energy -= self.energy_per_step
        time_step += 1
        
        # Out of energy or time
        if energy <= 0 or time_step >= self.max_time:
            if goal_reached:
                distance = abs(self.start[0] - y) + abs(self.start[1] - x)
            else:
                distance = abs(self.goal[0] - y) + abs(self.goal[1] - x)
            return state, self._calculate_reward(0, time_step, energy, False, False, False, distance, distance, goal_reached), True, {}        
        
        # Check if any moves are possible, so episode can end early if rover is stuck
        can_move_anywhere = False
        for check_action in range(4):
            is_valid, _, _ = self.calculate_energy_cost(y, x, check_action, last_action)
            if is_valid:
                can_move_anywhere = True
                break
        
        # End early if stuck
        if not can_move_anywhere:
            # Compute remaining distance to objective
            if goal_reached:
                distance = abs(self.start[0] - y) + abs(self.start[1] - x)
            else:
                distance = abs(self.goal[0] - y) + abs(self.goal[1] - x)
            return state, self._calculate_reward(0, time_step, 0, False, False, False, distance, distance, goal_reached), True, {'stuck': True}        
        
        is_valid, (dy, dx), energy_cost = self.calculate_energy_cost(y, x, action, last_action)
        
        if not is_valid:
            energy -= energy_cost
            next_state = (y, x, max(0, energy), time_step, visited_mask, -1, -1, goal_reached)
            if goal_reached:
                distance = abs(self.start[0] - y) + abs(self.start[1] - x)
            else:
                distance = abs(self.goal[0] - y) + abs(self.goal[1] - x)
            reward = self._calculate_reward(energy_cost, time_step, energy, False, False, False, distance, distance, goal_reached)
            done = energy <= 0 or time_step >= self.max_time
            return next_state, reward, done, {}
        
        # Compute old and new distances for reward
        if goal_reached:
            old_distance = abs(self.start[0] - y) + abs(self.start[1] - x)
        else:
            old_distance = abs(self.goal[0] - y) + abs(self.goal[1] - x)
        next_y, next_x = y + dy, x + dx
        energy -= energy_cost
        if goal_reached:
            new_distance = abs(self.start[0] - next_y) + abs(self.start[1] - next_x)
        else:
            new_distance = abs(self.goal[0] - next_y) + abs(self.goal[1] - next_x)
        
        if energy < 0:
            energy = 0
        
        new_visited_mask = visited_mask
        new_goal_reached = goal_reached
        goal_just_reached = False
        at_charging_station = False
        base_reached = False
        
        # Completed goal leg
        if not goal_reached and (next_y, next_x) == self.goal:
            goal_just_reached = True
            new_goal_reached = True
        
        # Completed return leg
        if goal_reached and (next_y, next_x) == self.start:
            base_reached = True
        
        # Add energy if at a charging station
        if (next_y, next_x) in self.charging_station_dict:
            station_idx = self.charging_station_dict[(next_y, next_x)]
            if (visited_mask & (1 << station_idx)) == 0:
                at_charging_station = True
        
        if at_charging_station and not goal_just_reached and not base_reached:
            station_idx = self.charging_station_dict[(next_y, next_x)]
            if (visited_mask & (1 << station_idx)) == 0:
                energy = min(self.max_energy, energy + 50)
                new_visited_mask |= (1 << station_idx)
        
        # Get next state and reward
        next_state = (next_y, next_x, energy, time_step, new_visited_mask, action, action, new_goal_reached)
        reward = self._calculate_reward(energy_cost, time_step, energy, goal_just_reached, base_reached, at_charging_station, old_distance, new_distance, new_goal_reached)
        done = base_reached or energy <= 0 or time_step >= self.max_time
        
        return next_state, reward, done, {'goal_reached': goal_just_reached, 'base_reached': base_reached, 'at_charging_station': at_charging_station, 'energy_used': energy_cost, 'energy_remaining': energy}
    
    def _calculate_reward(self, energy_cost, time_step, energy, goal_just_reached, base_reached, at_charging_station, old_distance, new_distance, goal_reached, overlap_penalty=0):
        """
        Calculates the reward of the current move.

        Parameters:
            energy_cost (float): The energy cost of the current move.
            time_step (int): The current time step of the move.
            energy (float): The amount of energy the rover has left.
            goal_just_reached (boolean): If the rover has reached the goal this turn.
            base_reached (boolean): If the rover has returned to the start after reaching the goal.
            at_charging_station (boolean): If the rover is at a charging station.
            old_distance (int): The rover's distance to the objective last turn.
            new_distance (int): The rover's distance to the objective this turn.
            goal_reached (boolean): If the rover has reached the goal yet.
            overlap_penalty (int): penalty for the rover retracing its steps.

        Returns:
            int: The reward of the current move.
        """
        base_bonus = 1500.0 if base_reached else 0
        goal_bonus = 1000.0 if goal_just_reached else 0
        time_penalty = 0.002 * time_step
        energy_penalty = 1.0 * energy_cost
        depletion_penalty = 300.0 if energy <= 0 else 0
        charging_bonus = 75.0 if at_charging_station else 0
        distance_reward = 1.0 * (old_distance - new_distance)

        return base_bonus + goal_bonus - time_penalty - energy_penalty + charging_bonus - depletion_penalty + distance_reward
    
    def get_observation(self, state):
        """
        Analyses the current state and converts it into a more easily readable dictionary

        Parameters:
            state (tuple): The rover's state, containing its position, energy, the current time, etc.
        
        Returns:
            dict: Contains state information in key-value pairs.
        """

        y, x, energy, time_step, visited_mask, _, _, goal_reached = state

        # Terrain information of the 5x5 area surrounding the rover 
        local_grid = np.zeros((5, 5))
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    local_grid[dy+2, dx+2] = self.terrain[ny, nx] / 10.0  
                else:
                    local_grid[dy+2, dx+2] = -1  
        
        # Get distance and direction to objective (start/goal depending on if goal already reached)
        if goal_reached:
            objective_dy, objective_dx = self.start[0] - y, self.start[1] - x
            objective_distance = abs(objective_dy) + abs(objective_dx)
            objective_direction = np.arctan2(objective_dy, objective_dx)
        else:
            objective_dy, objective_dx = self.goal[0] - y, self.goal[1] - x
            objective_distance = abs(objective_dy) + abs(objective_dx)
            objective_direction = np.arctan2(objective_dy, objective_dx)
        
        # Goal distance and direction
        goal_dy, goal_dx = self.goal[0] - y, self.goal[1] - x
        goal_distance = abs(goal_dy) + abs(goal_dx)
        goal_direction = np.arctan2(goal_dy, goal_dx)
        
        # Start distance and direction
        start_dy, start_dx = self.start[0] - y, self.start[1] - x
        start_distance = abs(start_dy) + abs(start_dx)
        start_direction = np.arctan2(start_dy, start_dx)
        
        # Get which charging stations were visited
        charging_info = []
        for i, (cs_y, cs_x) in enumerate(self.charging_stations):
            if (visited_mask & (1 << i)) == 0:  
                cs_dy, cs_dx = cs_y - y, cs_x - x
                charging_info.append({
                    'direction': np.arctan2(cs_dy, cs_dx),
                    'distance': abs(cs_dy) + abs(cs_dx),
                    'index': i
                })
        
        return {
            'local_grid': local_grid,
            'goal_direction': goal_direction,
            'goal_distance': goal_distance,
            'objective_direction': objective_direction,
            'objective_distance': objective_distance,
            'energy': energy,
            'time': time_step,
            'max_energy': self.max_energy,
            'start_direction': start_direction,
            'start_distance': start_distance,
            'charging_stations': charging_info,
            'position': (y, x),
            'goal_reached': goal_reached
        }