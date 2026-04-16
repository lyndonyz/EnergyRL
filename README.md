# EnergyRL: Energy Usage in Relation to Reinforcement Learning Project Setup
### Tool versions:
Python Verison: 3.10.11 <br>
numpy Version: 2.2.6 <br>
matplotlib Version: 3.10.8

---
### How to run project:
Depending on what you are on, you might need to run a virtual environment. <br>
Inside the directory run: `python3 -m venv venv` to create an virtual env.

After setting up your virtual environment, run it by using one of the following: <br>
`source venv/bin/activate` on MacOS <br>
`venv\Scripts\activate` on Windows <br>
<br>
You must now install both libraries used within this project to run it.<br>
Install numpy using: `pip install numpy`<br>
Install matplotlib using: `pip install matplotlib`<br>
<br>
You can then run:<br>
`python TestRoverMars.py (yes || Null)` to show training of rover on singluar map. Adding yes afterwards saves an image for each 100 episodes (80 total images). By default will show the final route, Avg reward per episode, Avg Energy per episode<br>
`python TestTrainedRover.py (yes || Null)` to show simulations of the trained rover on 50 random maps. Adding yes shows the most optimal route using A*. By default no optimal route will show. Will save 50 total images.

# Rover Energy & Mission Mechanics

This section describes the energy consumption, movement rules, and reward system for the Rover simulation.

Please note that these numbers are pulled roughly from external sources, and are not accurate to actual real world energy usage. They are similar enough, but not an exact model. This is due to energy usage being more of a factor on the model and design of the rover we are testing it on.

---

## Energy Consumption Breakdown

Energy is the most critical resource for the rover. Every action consumes energy, modified by momentum (turning) and terrain (slope).

### Base Costs 

These are just some costs to help standarize the energy usage throughout our project.

| Action | Energy Cost | Notes |
|--------|------------|-------|
| Idle Overhead (`energy_per_step`) | 0.1 | Consumed at the start of every step, even if no movement occurs. |
| Standard Move (`energy_move`) | 1.0 | Moving to an adjacent cell. |
| Failed Move / Stop (`energy_stop`) | 0.3 | Occurs if the rover hits a boundary or a slope too steep. |

### Turning & Momentum

Energy cost increases if the rover changes direction from its last action:

| Turn | Extra Cost | Example |
|------|------------|---------|
| Same Direction | 0.0 | Moving Up then Up |
| 90° Turn | 1.4 | Moving Up then Right |
| 180° Turn (U-turn) | 1.8 | Moving Up then Down |

### Terrain & Slope Modifiers

Energy cost is influenced by elevation changes (`Δe`) between the current and target cell. The reason behind this is to make our model really think about whether or not to use extra energy when moving up in elevation as it uses more energy when moving up in general based on slope. Another thing to note is that moving downhill in general reduces energy cost but stopping to turn uses more energy as the rover must maintain its position without falling over. All of these calculations roughly follow that logic.

#### Uphill Movement (`Δe > 0`)
Cost = Base Movement cost + (Δe * 0.15 * Multiplier)


The multiplier depends on the slope angle `θ` (calculated via arctan):

| Slope Angle (θ) | Multiplier Formula |
|-----------------|-----------------|
| θ ≤ 5° | 1.0 + (θ / 5.0) × 0.2 |
| 5° < θ ≤ 10° | 1.2 + ((θ - 5) / 5.0) × 0.3 |
| 10° < θ ≤ 15° | 1.5 + ((θ - 10) / 5.0) × 1.0 |
| θ > 15° | 2.5 + ((θ - 15) / 10.0) × 1.5 |

#### Downhill Movement (`Δe < 0`)

Downhill reduces energy cost (minimum multiplier of 0.3):

| Vertical Drop (`\|Δe\|`) | Multiplier Formula |
| :--- | :--- |
| ≤ 0.3 m | 0.9 - (`\|Δe\|` / 0.3) × 0.2 |
| 0.3 m < `\|Δe\|` ≤ 1.0 m | 0.8 - ( (`\|Δe\|` - 0.3) / 0.7 ) × 0.3 |
| > 1.0 m | 0.6 - ( (`\|Δe\|` - 1.0) / 1.0 ) × 0.3 |

---

## Time and Mission Constraints

- **Maximum Time**: Default 500 steps (600 in test scripts)  
- **Time Penalty**: -0.002 per step to encourage efficiency  
- **Mission End Conditions**:
  - Goal reached  
  - Energy hits 0  
  - Max time reached  
  - Rover is stuck (no valid moves due to steep slopes)

---

## Rover Mechanics

### State and Observation

The rover uses `get_observation` to perceive its surroundings:

- **Local Terrain**: 5x5 elevation grid centered on current position. This is to simulate a rovers roughly 2meter x 2meter view distence.  
- **Navigation**: Direction vectors to Goal, Start, and remaining Charging Stations  
- **Vitals**: Current energy and elapsed time  

### Recharging

- **Charging Stations**: 3 stations on the map  
- **Energy Boost**: +50 energy per station (Energy is capped at max 100)  
- **One-time Use**: Deactivated after visit (tracked with `visited_mask`)  

### Movement Constraints

- **Max Slope**: 2.0 m vertical difference as rovers wouldn't be able to make a distance like that. (`max_slope`)  
- **Energy Discretization**: Energy divided into 20-unit chunks for Q-learning  

---

## Two-Phase Mission Mechanics

The rover operates in two distinct phases:

1. **Phase 1 (Exploration)**: Navigate from start to goal location, visiting charging stations as needed
2. **Phase 2 (Return)**: After reaching goal, navigate back to base with intelligent charging to ensure survival

The rover transitions automatically between phases when the goal is reached. During Phase 2, the rover estimates required energy for return and seeks nearby charging stations if low on energy.

The rover estimates energy needs as `distance_to_base × 2.5` specificially when returning to account for charging station detours. If estimated energy falls below this threshold, the rover seeks the nearest charging station before continuing toward base.

---

## Reward Logic

The rover is guided by a weighted reward system:

| Event | Reward |
|-------|--------|
| Base Return | +1500 |
| Goal Reached | +1000 |
| Charging | +75 |
| Distance Progress | +1.0 × decrease in distance to objective |
| Energy Used | -1.0 × energy consumed this step |
| Energy Depletion | -300 |

## Next Steps that have been Implemented
* Improved rover decision-making strategy
  * Smarter planning of when to recharge to avoid unnecessary detours
  * Better handling of energy vs distance trade-offs
* Extend to multi-objective, multi-goal planning
  * Balance energy usage, distance, and time across several targets


Unfortunately due to the complexity of adding realistic physics and energy costs along with our time constrant, we didn't get to implement it for our final report.