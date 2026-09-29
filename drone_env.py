"""
drone_env.py
------------
Custom Gymnasium environment for Battery-Aware Drone Delivery in Stochastic
Wind Conditions.

Subclasses gymnasium.Env. Implements discrete state space encoding, discrete
action dynamics, battery drainage, stochastic wind perturbation, urgent deadline
bucketing, and reward mechanics.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import config


def encode_state(x: int, y: int, battery: int, carrying: int, delivered_mask: int, urgent_bucket: int) -> int:
    """
    Bijective mixed-radix encoding of the 6 state variables into a single integer.
    Radix dimensions:
      - x: 8 (0..7)
      - y: 8 (0..7)
      - battery: MAX_BATTERY + 1 (0..MAX_BATTERY)
      - carrying: 2 (0 or 1)
      - delivered_mask: 8 (0..7, representing 3 binary flags)
      - urgent_bucket: 3 (0..2, coarse time-to-deadline)
    """
    idx = x
    idx = idx * config.GRID_SIZE + y
    idx = idx * (config.MAX_BATTERY + 1) + battery
    idx = idx * 2 + carrying
    idx = idx * 8 + delivered_mask
    idx = idx * config.URGENT_BUCKETS + urgent_bucket
    return int(idx)


def decode_state(state_idx: int) -> tuple[int, int, int, int, int, int]:
    """
    Decodes an integer state index back into its 6 component variables:
    (x, y, battery, carrying, delivered_mask, urgent_bucket).
    """
    idx = int(state_idx)
    urgent_bucket = idx % config.URGENT_BUCKETS
    idx //= config.URGENT_BUCKETS
    delivered_mask = idx % 8
    idx //= 8
    carrying = idx % 2
    idx //= 2
    battery = idx % (config.MAX_BATTERY + 1)
    idx //= (config.MAX_BATTERY + 1)
    y = idx % config.GRID_SIZE
    x = idx // config.GRID_SIZE
    return x, y, battery, carrying, delivered_mask, urgent_bucket


class DroneDeliveryEnv(gym.Env):
    """
    Gymnasium environment simulating a delivery drone navigating an 8x8 grid.
    Features:
      - Depot pickup and 3 drop-off destinations (D3 has an urgent deadline).
      - Recharging stations to counteract battery depletion.
      - Stochastic wind zones that deflect flight paths sideways with prob 0.2.
      - Payload-dependent battery consumption.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, render_mode: str = "ansi"):
        super().__init__()
        self.render_mode = render_mode

        # Action space: 6 discrete actions (UP, DOWN, LEFT, RIGHT, PICKUP/DROPOFF, CHARGE)
        self.action_space = spaces.Discrete(config.NUM_ACTIONS)

        # Total number of unique states in the discrete state space
        self.num_states = (
            config.GRID_SIZE
            * config.GRID_SIZE
            * (config.MAX_BATTERY + 1)
            * 2
            * 8
            * config.URGENT_BUCKETS
        )
        self.observation_space = spaces.Discrete(self.num_states)

        # Grid configuration cached for convenience
        self.depot = config.DEPOT
        self.chargers = set(config.CHARGERS)
        self.dropoffs = list(config.DROPOFFS)
        self.wind_zones = set(config.WIND_ZONES)
        self.urgent_idx = config.URGENT_DROPOFF_IDX
        self.all_delivered_mask = (1 << len(self.dropoffs)) - 1  # 0b111 = 7

        # Internal state variables
        self.x = 0
        self.y = 0
        self.battery = config.MAX_BATTERY
        self.carrying = 0
        self.delivered_mask = 0
        self.step_count = 0
        self.urgent_missed_penalized = False

        # Random generator
        self.np_random = None
        self.seed(config.SEED)

    def seed(self, seed: int | None = None):
        """Seed the environment's internal random number generator."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _get_urgent_bucket(self) -> int:
        """
        Discretizes remaining time for urgent package D3 into 3 coarse buckets:
          - Bucket 0: Safe (more than half the deadline remains, or already delivered)
          - Bucket 1: Warning (less than or equal to half deadline remaining)
          - Bucket 2: Expired (step_count exceeded URGENT_DEADLINE)
        """
        if self.delivered_mask & (1 << self.urgent_idx):
            return 0  # Mission accomplished for D3; no longer urgent

        rem = config.URGENT_DEADLINE - self.step_count
        if rem > config.URGENT_DEADLINE // 2:
            return 0
        elif rem > 0:
            return 1
        return 2

    def _get_obs(self) -> int:
        """Returns the encoded integer state observation."""
        return encode_state(
            self.x,
            self.y,
            self.battery,
            self.carrying,
            self.delivered_mask,
            self._get_urgent_bucket(),
        )

    def _get_info(self) -> dict:
        """Returns a human-readable dictionary describing the state."""
        deliv_flags = [bool(self.delivered_mask & (1 << i)) for i in range(len(self.dropoffs))]
        return {
            "x": self.x,
            "y": self.y,
            "pos": (self.x, self.y),
            "battery": self.battery,
            "carrying": bool(self.carrying),
            "delivered": deliv_flags,
            "delivered_count": sum(deliv_flags),
            "delivered_mask": self.delivered_mask,
            "urgent_bucket": self._get_urgent_bucket(),
            "urgent_bucket_desc": ["Safe", "Warning", "Expired"][self._get_urgent_bucket()],
            "steps": self.step_count,
            "urgent_met_on_time": bool(self.delivered_mask & (1 << self.urgent_idx)) and (not self.urgent_missed_penalized),
            "all_delivered": self.delivered_mask == self.all_delivered_mask,
        }

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[int, dict]:
        """Resets the environment to the initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.x, self.y = self.depot
        self.battery = config.MAX_BATTERY
        self.carrying = 0
        self.delivered_mask = 0
        self.step_count = 0
        self.urgent_missed_penalized = False

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """
        Executes one environmental step.
        Returns: (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        reward = config.REWARD_STEP_PENALTY  # Baseline step penalty (-1.0)
        terminated = False
        truncated = False

        # One-off penalty when urgent deadline crosses without D3 delivered
        if not (self.delivered_mask & (1 << self.urgent_idx)):
            if self.step_count == config.URGENT_DEADLINE and not self.urgent_missed_penalized:
                reward += config.REWARD_URGENT_MISSED
                self.urgent_missed_penalized = True

        # Determine battery drain for movement actions
        drain = config.LOADED_DRAIN if self.carrying else config.EMPTY_DRAIN

        # ======================================================================
        # Action 0..3: Directional Moves
        # ======================================================================
        if action in [config.ACTION_UP, config.ACTION_DOWN, config.ACTION_LEFT, config.ACTION_RIGHT]:
            nx, ny = self.x, self.y
            if action == config.ACTION_UP:
                ny = min(config.GRID_SIZE - 1, self.y + 1)
            elif action == config.ACTION_DOWN:
                ny = max(0, self.y - 1)
            elif action == config.ACTION_LEFT:
                nx = max(0, self.x - 1)
            elif action == config.ACTION_RIGHT:
                nx = min(config.GRID_SIZE - 1, self.x + 1)

            # Stochastic wind deflection if landing in or moving inside a wind zone
            if (nx, ny) in self.wind_zones and (self.np_random.random() < config.WIND_PUSH_PROB):
                # Perpendicular push:
                # Vertical moves (UP/DOWN) pushed horizontally (LEFT/RIGHT)
                # Horizontal moves (LEFT/RIGHT) pushed vertically (UP/DOWN)
                if action in [config.ACTION_UP, config.ACTION_DOWN]:
                    push = int(self.np_random.choice([-1, 1]))
                    nx = min(max(0, nx + push), config.GRID_SIZE - 1)
                else:
                    push = int(self.np_random.choice([-1, 1]))
                    ny = min(max(0, ny + push), config.GRID_SIZE - 1)

            self.x, self.y = nx, ny
            self.battery -= drain

        # ======================================================================
        # Action 4: PICKUP / DROPOFF
        # ======================================================================
        elif action == config.ACTION_PICKUP_DROPOFF:
            useful_action = False

            # Pickup at Depot
            if (self.x, self.y) == self.depot and self.carrying == 0 and self.delivered_mask != self.all_delivered_mask:
                self.carrying = 1
                useful_action = True
                reward += config.REWARD_PICKUP

            # Delivery at an undelivered dropoff cell
            elif self.carrying == 1:
                for i, drop_pos in enumerate(self.dropoffs):
                    if (self.x, self.y) == drop_pos and not (self.delivered_mask & (1 << i)):
                        self.delivered_mask |= (1 << i)
                        self.carrying = 0
                        useful_action = True

                        if i == self.urgent_idx:
                            # Urgent delivery
                            if self.step_count <= config.URGENT_DEADLINE:
                                reward += config.REWARD_URGENT_ON_TIME
                            else:
                                reward += config.REWARD_URGENT_LATE
                        else:
                            # Normal delivery
                            reward += config.REWARD_NORMAL_DELIVERY
                        break

            if not useful_action:
                reward += config.REWARD_USELESS_ACTION

            # Picking up or dropping off consumes empty move drain (hovering)
            self.battery -= config.EMPTY_DRAIN

        # ======================================================================
        # Action 5: CHARGE
        # ======================================================================
        elif action == config.ACTION_CHARGE:
            if (self.x, self.y) in self.chargers:
                if self.battery < config.MAX_BATTERY:
                    self.battery = min(config.MAX_BATTERY, self.battery + config.CHARGE_RATE)
                    reward += config.REWARD_CHARGE_STEP
                else:
                    # Already full; slight useless penalty
                    reward += config.REWARD_USELESS_ACTION
            else:
                # Attempted charging away from a designated station
                reward += config.REWARD_USELESS_ACTION
                self.battery -= config.EMPTY_DRAIN

        # ======================================================================
        # Termination & Truncation Checks
        # ======================================================================
        # 1. Battery depleted: catastrophic failure
        if self.battery <= 0:
            reward += config.REWARD_BATTERY_DEATH
            self.battery = 0
            terminated = True

        # 2. All 3 dropoffs successfully delivered: mission complete!
        elif self.delivered_mask == self.all_delivered_mask:
            terminated = True

        # 3. Maximum steps reached: truncated
        elif self.step_count >= config.MAX_STEPS:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self) -> str:
        """
        Renders the grid environment as ASCII text.
        Symbols:
          - D : Drone (D* if carrying package)
          - H : Home Depot
          - C : Charging station
          - 1, 2, 3: Drop-offs D1, D2, D3
          - W : Wind zone
          - . : Empty cell
        """
        grid = [["." for _ in range(config.GRID_SIZE)] for _ in range(config.GRID_SIZE)]

        # Place Wind zones
        for wx, wy in self.wind_zones:
            grid[wy][wx] = "W"

        # Place Chargers
        for cx, cy in self.chargers:
            grid[cy][cx] = "C"

        # Place Depot
        dx, dy = self.depot
        grid[dy][dx] = "H"

        # Place Dropoffs
        for i, (px, py) in enumerate(self.dropoffs):
            is_done = bool(self.delivered_mask & (1 << i))
            grid[py][px] = "✓" if is_done else str(i + 1)

        # Place Drone
        drone_sym = "D*" if self.carrying else "D"
        grid[self.y][self.x] = drone_sym

        # Build ASCII string (y=7 at top, y=0 at bottom)
        border = "+" + "---+" * config.GRID_SIZE
        lines = [border]
        for y in reversed(range(config.GRID_SIZE)):
            row = "|"
            for x in range(config.GRID_SIZE):
                cell = grid[y][x]
                row += f"{cell:^3}|"
            lines.append(row)
            lines.append(border)

        deliv_status = [
            f"D{i+1}:{'DONE' if (self.delivered_mask & (1 << i)) else 'PENDING'}"
            for i in range(len(self.dropoffs))
        ]
        info_lines = [
            f"Step: {self.step_count}/{config.MAX_STEPS} | Pos: ({self.x}, {self.y}) | "
            f"Battery: {self.battery}/{config.MAX_BATTERY} | Carrying: {bool(self.carrying)}",
            f"Deliveries: [{', '.join(deliv_status)}] | Urgent Bucket: {['Safe', 'Warning', 'Expired'][self._get_urgent_bucket()]}",
        ]

        output = "\n".join(info_lines) + "\n" + "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output


def random_policy_baseline(episodes: int = 100, seed: int = config.SEED) -> dict:
    """
    Evaluates a uniform random action policy baseline on DroneDeliveryEnv.
    Used to establish baseline performance and prove the environment is sound
    and the task is non-trivial.
    """
    env = DroneDeliveryEnv()
    rewards = []
    successes = []
    deliveries = []
    steps_list = []
    battery_deaths = 0

    for ep in range(episodes):
        ep_seed = seed + ep
        obs, info = env.reset(seed=ep_seed)
        done = False
        ep_reward = 0.0

        while not done:
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            done = term or trunc

        rewards.append(ep_reward)
        deliveries.append(info["delivered_count"])
        successes.append(1 if info["all_delivered"] else 0)
        steps_list.append(info["steps"])
        if info["battery"] == 0:
            battery_deaths += 1

    summary = {
        "episodes": episodes,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes) * 100.0),
        "avg_deliveries": float(np.mean(deliveries)),
        "avg_steps": float(np.mean(steps_list)),
        "battery_death_rate": float(battery_deaths / episodes * 100.0),
    }

    print("\n" + "=" * 60)
    print("           RANDOM POLICY BASELINE RESULTS")
    print("=" * 60)
    print(f"Episodes Evaluated    : {summary['episodes']}")
    print(f"Average Return        : {summary['avg_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"Success Rate (all 3)  : {summary['success_rate']:.1f}%")
    print(f"Average Deliveries    : {summary['avg_deliveries']:.2f} / 3.0")
    print(f"Average Steps         : {summary['avg_steps']:.1f}")
    print(f"Battery Exhaustion Rate: {summary['battery_death_rate']:.1f}%")
    print("=" * 60 + "\n")

    return summary


if __name__ == "__main__":
    # Smoke test environment and verify encoding
    print("Testing DroneDeliveryEnv...")
    env = DroneDeliveryEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    print(f"Initial encoded state: {obs}, info: {info}")
    decoded = decode_state(obs)
    print(f"Decoded state matches: {decoded == (info['x'], info['y'], info['battery'], int(info['carrying']), info['delivered_mask'], info['urgent_bucket'])}")
    env.render()

    print("\nRunning random policy baseline...")
    random_policy_baseline(episodes=100)
