"""
base_agent.py
-------------
Base tabular Reinforcement Learning agent.
Provides shared functionality for Q-learning and SARSA:
  - NumPy Q-table storage
  - Epsilon-greedy action selection
  - Epsilon decay
  - Q-table serialization (save/load)
"""

import numpy as np
import config


class BaseAgent:
    """
    Abstract tabular RL agent using an exact NumPy Q-table.
    Subclasses implement the specific Bellman temporal-difference update rule.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int = config.NUM_ACTIONS,
        alpha: float = config.ALPHA,
        gamma: float = config.GAMMA,
        epsilon_start: float = config.EPSILON_START,
        epsilon_min: float = config.EPSILON_MIN,
        epsilon_decay: float = config.EPSILON_DECAY,
        seed: int = config.SEED,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = np.random.default_rng(seed)

        # Tabular Q-value storage: shape (num_states, num_actions)
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float32)

    def select_action(self, state: int, greedy: bool = False) -> tuple[int, bool]:
        """
        Selects an action using an epsilon-greedy policy.

        Parameters:
          state: Encoded integer state index
          greedy: If True, bypasses exploration and takes argmax_a Q(s, a)

        Returns:
          (action, is_greedy): Chosen action index and boolean indicating whether
                               the choice was greedy (exploitative) or exploratory.
        """
        if not greedy and (self.rng.random() < self.epsilon):
            # Exploration: uniform random action
            action = int(self.rng.integers(0, self.num_actions))
            return action, False

        # Exploitation: break ties arbitrarily among actions with max Q-value
        q_row = self.q_table[state]
        max_q = np.max(q_row)
        best_actions = np.flatnonzero(q_row == max_q)
        if len(best_actions) > 1:
            action = int(self.rng.choice(best_actions))
        else:
            action = int(best_actions[0])
        return action, True

    def decay_epsilon(self) -> float:
        """Decays epsilon by multiplicative factor down to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, filepath: str) -> None:
        """Saves the NumPy Q-table to disk (.npy format)."""
        np.save(filepath, self.q_table)

    def load(self, filepath: str) -> None:
        """Loads a pre-trained NumPy Q-table from disk."""
        self.q_table = np.load(filepath)
