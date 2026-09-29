"""
evaluate.py
-----------
Evaluation script for trained Q-Learning and SARSA policies on DroneDeliveryEnv.

Runs a purely greedy evaluation (epsilon = 0.0) over 500 episodes to measure:
  - Average return (cumulative reward)
  - Task success rate (all 3 packages delivered)
  - Average deliveries completed (out of 3)
  - Average steps per episode
  - Urgent delivery on-time success rate
  - Battery exhaustion rate

Structured so that Part II metrics (regret, sensitivity analysis, sample complexity)
can be seamlessly incorporated.

Usage:
    python evaluate.py --algo qlearning
    python evaluate.py --algo sarsa
    python evaluate.py --algo qlearning --episodes 500
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

import config
from drone_env import DroneDeliveryEnv
from q_agent import QLearningAgent
from sarsa_agent import SARSAAgent


def evaluate_policy(algo: str = "qlearning", episodes: int = 500, seed: int = 9999):
    """
    Evaluates a trained Q-table using a greedy policy (epsilon = 0.0).
    """
    model_path = f"results/q_table_{algo}.npy"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print(f"Please train the agent first using: python train.py --algo {algo}")
        sys.exit(1)

    print("=" * 70)
    print(f"     POLICY EVALUATION: {algo.upper()} ({episodes} EPISODES, GREEDY)")
    print("=" * 70)

    # 1. Initialize environment
    env = DroneDeliveryEnv()
    env.seed(seed)

    # 2. Load agent
    if algo == "qlearning":
        agent = QLearningAgent(num_states=env.num_states, num_actions=config.NUM_ACTIONS)
    elif algo == "sarsa":
        agent = SARSAAgent(num_states=env.num_states, num_actions=config.NUM_ACTIONS)
    else:
        raise ValueError(f"Unknown algorithm '{algo}'")

    agent.load(model_path)
    agent.epsilon = 0.0  # Strict exploitation

    rewards = []
    successes = []
    deliveries = []
    steps_list = []
    urgent_met = []
    battery_deaths = 0

    # 3. Evaluation loop
    for ep in range(episodes):
        ep_seed = seed + ep
        state, info = env.reset(seed=ep_seed)
        done = False
        ep_r = 0.0

        while not done:
            action, _ = agent.select_action(state, greedy=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_r += reward
            done = terminated or truncated
            state = next_state

        rewards.append(ep_r)
        successes.append(1 if info["all_delivered"] else 0)
        deliveries.append(info["delivered_count"])
        steps_list.append(info["steps"])
        urgent_met.append(1 if info["urgent_met_on_time"] else 0)
        if info["battery"] == 0:
            battery_deaths += 1

    # 4. Compute statistics
    avg_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    succ_rate = float(np.mean(successes) * 100.0)
    avg_deliv = float(np.mean(deliveries))
    avg_steps = float(np.mean(steps_list))
    std_steps = float(np.std(steps_list))
    urgent_rate = float(np.mean(urgent_met) * 100.0)
    death_rate = float(battery_deaths / episodes * 100.0)

    print(f"Algorithm                  : {algo.upper()}")
    print(f"Model Path                 : {model_path}")
    print(f"Episodes Evaluated         : {episodes}")
    print(f"Mean Cumulative Return     : {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"3-Package Success Rate     : {succ_rate:.1f}%")
    print(f"Average Deliveries Done    : {avg_deliv:.2f} / 3.0")
    print(f"Average Steps Taken        : {avg_steps:.1f} ± {std_steps:.1f}")
    print(f"Urgent Package (D3) On-Time: {urgent_rate:.1f}%")
    print(f"Battery Death Rate         : {death_rate:.1f}%")
    print("=" * 70 + "\n")

    return {
        "algo": algo,
        "episodes": episodes,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "success_rate": succ_rate,
        "avg_deliveries": avg_deliv,
        "avg_steps": avg_steps,
        "urgent_rate": urgent_rate,
        "death_rate": death_rate,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained RL policy")
    parser.add_argument("--algo", type=str, default="qlearning", choices=["qlearning", "sarsa"], help="Algorithm to evaluate")
    parser.add_argument("--episodes", type=int, default=500, help="Number of greedy evaluation episodes")
    parser.add_argument("--seed", type=int, default=9999, help="Random seed for evaluation")
    args = parser.parse_args()

    evaluate_policy(algo=args.algo, episodes=args.episodes, seed=args.seed)
