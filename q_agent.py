"""
q_agent.py
----------
Tabular Q-Learning Agent (Off-Policy Temporal Difference Learning).

Update rule:
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

The agent updates its estimates toward the greedy optimal action at next_state,
regardless of what action is actually executed next (off-policy).
"""

import numpy as np
from base_agent import BaseAgent
import config


class QLearningAgent(BaseAgent):
    """
    Q-Learning agent implementing off-policy TD(0) control.
    """

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> dict:
        """
        Performs one Q-learning update step.

        Parameters:
          state: Current state index s
          action: Action taken a
          reward: Reward received r
          next_state: Resulting state index s'
          terminated: Whether the transition reached a terminal state

        Returns:
          dict with update diagnostics (old_q, target, new_q, td_error)
        """
        old_q = float(self.q_table[state, action])

        # Bellman optimality target:
        # If terminal, no future discounted return can be collected.
        if terminated:
            td_target = float(reward)
        else:
            best_next_q = float(np.max(self.q_table[next_state]))
            td_target = float(reward + self.gamma * best_next_q)

        td_error = float(td_target - old_q)
        new_q = float(old_q + self.alpha * td_error)

        # In-place update to NumPy array
        self.q_table[state, action] = new_q

        return {
            "old_q": old_q,
            "target": td_target,
            "new_q": new_q,
            "td_error": td_error,
        }


if __name__ == "__main__":
    # Smoke test initialization and update
    agent = QLearningAgent(num_states=100, num_actions=config.NUM_ACTIONS)
    diag = agent.update(state=0, action=4, reward=10.0, next_state=1, terminated=False)
    print("Q-Agent smoke test update:", diag)
    assert agent.q_table[0, 4] == diag["new_q"]
    print("Q-Agent successfully verified.")
