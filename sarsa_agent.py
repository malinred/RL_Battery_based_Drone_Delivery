"""
sarsa_agent.py
--------------
Tabular SARSA Agent (On-Policy Temporal Difference Learning).

Update rule:
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * Q(s', a') - Q(s, a) ]

The agent updates its estimates using the action a' chosen by the current
behavior policy (epsilon-greedy) at next_state (on-policy).
"""

from base_agent import BaseAgent
import config


class SARSAAgent(BaseAgent):
    """
    SARSA agent implementing on-policy TD(0) control.
    """

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        terminated: bool,
    ) -> dict:
        """
        Performs one SARSA update step.

        Parameters:
          state: Current state index s
          action: Action taken a
          reward: Reward received r
          next_state: Resulting state index s'
          next_action: Action selected for execution at s' (a')
          terminated: Whether the transition reached a terminal state

        Returns:
          dict with update diagnostics (old_q, target, new_q, td_error)
        """
        old_q = float(self.q_table[state, action])

        # On-policy Bellman target:
        if terminated:
            td_target = float(reward)
        else:
            next_q = float(self.q_table[next_state, next_action])
            td_target = float(reward + self.gamma * next_q)

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
    agent = SARSAAgent(num_states=100, num_actions=config.NUM_ACTIONS)
    diag = agent.update(state=0, action=4, reward=10.0, next_state=1, next_action=2, terminated=False)
    print("SARSA-Agent smoke test update:", diag)
    assert agent.q_table[0, 4] == diag["new_q"]
    print("SARSA-Agent successfully verified.")
