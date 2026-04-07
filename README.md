# EnergyRL: Energy Usage in Relation to Reinforcement Learning Project
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
`python TestRoverMars.py (yes || Null)` to show training of rover on singluar map. Adding yes afterwards saves an image for each 100 episodes (50 total images). By default will show the final route, Avg reward per episode, Avg Energy per episode<br>
`python TestTrainedRover.py (yes || Null)` to show simulations of the trained rover on 50 random maps. Adding yes shows the most optimal route using A*. By default no optimal route will show. Will save 50 total images.

# Rover Energy & Mission Mechanics

This document describes the energy consumption, movement rules, and reward system for the Mars Rover simulation.

---

## Energy Consumption Breakdown

Energy is the most critical resource for the rover. Every action consumes energy, modified by momentum (turning) and terrain (slope).

### 1. Base Costs

| Action | Energy Cost | Notes |
|--------|------------|-------|
| Idle Overhead (`energy_per_step`) | 0.1 | Consumed at the start of every step, even if no movement occurs. |
| Standard Move (`energy_move`) | 1.0 | Moving to an adjacent cell. |
| Failed Move / Stop (`energy_stop`) | 0.3 | Occurs if the rover hits a boundary or a slope too steep. |

### 2. Turning & Momentum

Energy cost increases if the rover changes direction from its last action:

| Turn | Extra Cost | Example |
|------|------------|---------|
| Same Direction | 0.0 | Moving Up then Up |
| 90° Turn | 1.4 | Moving Up then Right |
| 180° Turn (U-turn) | 1.8 | Moving Up then Down |

### 3. Terrain & Slope Modifiers

Energy cost is influenced by elevation changes (`Δe`) between the current and target cell.

#### Uphill Movement (`Δe > 0`)
Cost = Base Move + (Δe * 0.15 * Multiplier)


The multiplier depends on the slope angle `θ` (calculated via arctan):

| Slope Angle (θ) | Multiplier Formula |
|-----------------|-----------------|
| θ ≤ 5° | 1.0 + (θ / 5.0) × 0.2 |
| 5° < θ ≤ 10° | 1.2 + ((θ - 5) / 5.0) × 0.3 |
| 10° < θ ≤ 15° | 1.5 + ((θ - 10) / 5.0) × 1.0 |
| θ > 15° | 2.5 + ((θ - 15) / 10.0) × 1.5 |

#### Downhill Movement (`Δe < 0`)

Downhill reduces energy cost (minimum multiplier of 0.3):

| Vertical Drop (|Δe|) | Multiplier Formula |
|-----------------|-----------------|
| ≤ 0.3 m | 0.9 - (|Δe| / 0.3) × 0.2 |
| 0.3 m < |Δe| ≤ 1.0 m | 0.8 - ((|Δe| - 0.3) / 0.7) × 0.3 |
| > 1.0 m | 0.6 - ((|Δe| - 1.0) / 1.0) × 0.3 |

---

## Time and Mission Constraints

- **Maximum Time**: Default 500 steps (600 in test scripts)  
- **Time Penalty**: -0.01 per step to encourage efficiency  
- **Mission End Conditions**:
  - Goal reached  
  - Energy hits 0  
  - Max time reached  
  - Rover is stuck (no valid moves due to steep slopes)

---

## Rover Mechanics

### State and Observation

The rover uses `get_observation` to perceive its surroundings:

- **Local Terrain**: 5x5 elevation grid centered on current position  
- **Navigation**: Direction vectors to Goal, Start, and remaining Charging Stations  
- **Vitals**: Current energy and elapsed time  

### Recharging

- **Charging Stations**: 3 stations on the map  
- **Energy Boost**: +50 energy per station (capped at max 100)  
- **One-time Use**: Deactivated after visit (tracked with `visited_mask`)  

### Movement Constraints

- **Max Slope**: 2.0 m vertical difference (`max_slope`)  
- **Energy Discretization**: Energy divided into 20-unit chunks for Q-learning  

---

## Reward Logic

The rover is guided by a weighted reward system:

| Event | Reward |
|-------|--------|
| Goal Reached | +1000 |
| Charging | +75 |
| Distance Progress | +0.5 × decrease in distance to goal |
| Energy Used | -2.0 × energy consumed this step |
| Energy Depletion | -300 |
