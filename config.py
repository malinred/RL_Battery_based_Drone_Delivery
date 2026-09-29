"""
config.py
---------
Central configuration for the Battery-Aware Drone Delivery RL project.
All environment constants, rewards, and agent hyperparameters are defined
here to allow easy experimentation and transparency during viva examination.
"""

# ==============================================================================
# 1. GRID WORLD LAYOUT
# ==============================================================================
GRID_SIZE = 8  # 8x8 grid (x: 0..7, y: 0..7)

# Coordinates follow standard Cartesian plane:
# (0, 0) is the bottom-left corner; x increases rightwards, y increases upwards.
DEPOT = (0, 0)                          # Package pickup hub
CHARGERS = [(0, 4), (6, 0)]             # Charging stations C1, C2
DROPOFFS = [(7, 7), (7, 2), (2, 6)]     # Delivery locations D1, D2, D3
DROPOFF_NAMES = ["D1 (7,7)", "D2 (7,2)", "D3 [Urgent] (2,6)"]
URGENT_DROPOFF_IDX = 2                  # Index of D3 (urgent delivery)

# Wind zones: 2x2 central cluster
# Entering or moving inside causes a 20% chance of sideways push
WIND_ZONES = [(3, 3), (3, 4), (4, 3), (4, 4)]
WIND_PUSH_PROB = 0.20                   # Probability of stochastic deflection

# ==============================================================================
# 2. ACTION SPACE
# ==============================================================================
ACTION_UP = 0              # Move North (+y)
ACTION_DOWN = 1            # Move South (-y)
ACTION_LEFT = 2            # Move West (-x)
ACTION_RIGHT = 3           # Move East (+x)
ACTION_PICKUP_DROPOFF = 4  # Pick up at depot or deliver at target dropoff
ACTION_CHARGE = 5          # Recharge battery at a charging station

NUM_ACTIONS = 6

ACTION_NAMES = {
    ACTION_UP: "UP",
    ACTION_DOWN: "DOWN",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
    ACTION_PICKUP_DROPOFF: "PICKUP/DROPOFF",
    ACTION_CHARGE: "CHARGE",
}

# ==============================================================================
# 3. BATTERY & EPISODE LIMITS
# ==============================================================================
MAX_BATTERY = 30           # Discrete battery capacity (levels 0..30)
EMPTY_DRAIN = 1            # Drain per move / non-charging step when empty
LOADED_DRAIN = 2           # Drain per move when carrying a package
CHARGE_RATE = 2            # Battery units restored per CHARGE step at a station

URGENT_DEADLINE = 60       # Step count deadline for delivering urgent package D3
URGENT_BUCKETS = 3         # Coarse timer buckets: 0=Safe, 1=Warning, 2=Expired
MAX_STEPS = 200            # Maximum steps per episode before truncation

# ==============================================================================
# 4. REWARD STRUCTURE
# ==============================================================================
# Primary task rewards
REWARD_NORMAL_DELIVERY = 50.0       # Successful delivery of normal packages (D1, D2)
REWARD_URGENT_ON_TIME = 100.0       # Successful delivery of D3 before deadline
REWARD_URGENT_LATE = 20.0           # Delivery of D3 after deadline has expired
REWARD_STEP_PENALTY = -1.0          # Time penalty incurred per step
REWARD_URGENT_MISSED = -30.0        # One-off penalty when deadline is crossed without D3
REWARD_BATTERY_DEATH = -100.0       # Catastrophic failure penalty if battery hits 0
REWARD_USELESS_ACTION = -2.0        # Penalty for invalid action (charge away from station, etc.)

# Subgoal guidance rewards (tuned to enable tabular exploration across multi-stage horizon)
REWARD_PICKUP = 20.0                # Intermediate bonus for loading a needed package at depot
REWARD_CHARGE_STEP = 1.2            # Incentive for recharging at a station when < max (net +0.2 above step penalty)

# ==============================================================================
# 5. RL AGENT HYPERPARAMETERS
# ==============================================================================
ALPHA = 0.15               # Learning rate (step size for temporal difference updates)
GAMMA = 0.98               # Discount factor (prioritizes long-term mission completion)
EPSILON_START = 1.0        # Initial exploration probability
EPSILON_MIN = 0.03         # Minimum exploration probability
EPSILON_DECAY = 0.99985    # Multiplicative decay rate per episode (smooth over 30k episodes)
DEFAULT_EPISODES = 30000   # Number of training episodes
SEED = 42                  # Global random seed for reproducible experiments
