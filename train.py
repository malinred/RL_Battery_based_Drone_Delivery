"""
train.py
--------
Training script for Q-Learning and SARSA agents on DroneDeliveryEnv.

Usage:
    python train.py --algo qlearning
    python train.py --algo sarsa
    python train.py --algo qlearning --episodes 30000

Features:
  - Trains tabular Q-learning or SARSA agents.
  - Logs per-episode metrics (reward, steps, deliveries, success, epsilon) to CSV.
  - Periodic console progress reporting every 1,000 episodes.
  - Serializes trained Q-table to results/q_table_<algo>.npy.
  - Plots and saves comprehensive learning curves to results/plots/learning_curve_<algo>.png.
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from drone_env import DroneDeliveryEnv
from q_agent import QLearningAgent
from sarsa_agent import SARSAAgent


def train(algo: str = "qlearning", episodes: int = config.DEFAULT_EPISODES, seed: int = config.SEED):
    """
    Executes the training loop for the chosen RL algorithm.
    """
    print("=" * 70)
    print(f"Starting Training: Algorithm = {algo.upper()}, Episodes = {episodes}, Seed = {seed}")
    print("=" * 70)

    # 1. Initialize environment
    env = DroneDeliveryEnv()
    env.seed(seed)
    np.random.seed(seed)

    # 2. Instantiate agent
    if algo.lower() == "qlearning":
        agent = QLearningAgent(
            num_states=env.num_states,
            num_actions=config.NUM_ACTIONS,
            alpha=config.ALPHA,
            gamma=config.GAMMA,
            epsilon_start=config.EPSILON_START,
            epsilon_min=config.EPSILON_MIN,
            epsilon_decay=config.EPSILON_DECAY,
            seed=seed,
        )
    elif algo.lower() == "sarsa":
        agent = SARSAAgent(
            num_states=env.num_states,
            num_actions=config.NUM_ACTIONS,
            alpha=config.ALPHA,
            gamma=config.GAMMA,
            epsilon_start=config.EPSILON_START,
            epsilon_min=config.EPSILON_MIN,
            epsilon_decay=config.EPSILON_DECAY,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm '{algo}'. Choose 'qlearning' or 'sarsa'.")

    # Logging records
    records = []
    start_time = time.time()

    # 3. Main episode loop
    for ep in range(1, episodes + 1):
        ep_seed = seed + ep
        state, info = env.reset(seed=ep_seed)
        done = False
        ep_reward = 0.0

        if algo == "sarsa":
            action, _ = agent.select_action(state, greedy=False)

        while not done:
            if algo == "qlearning":
                action, _ = agent.select_action(state, greedy=False)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Q-learning off-policy update
                agent.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    terminated=terminated,
                )
                state = next_state

            elif algo == "sarsa":
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if not done:
                    next_action, _ = agent.select_action(next_state, greedy=False)
                else:
                    next_action = 0

                # SARSA on-policy update
                agent.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    next_action=next_action,
                    terminated=terminated,
                )
                state = next_state
                action = next_action

            ep_reward += reward

        current_eps = agent.decay_epsilon()

        # Record episode metrics
        records.append({
            "episode": ep,
            "reward": ep_reward,
            "steps": info["steps"],
            "deliveries": info["delivered_count"],
            "success": int(info["all_delivered"]),
            "urgent_on_time": int(info["urgent_met_on_time"]),
            "battery_left": info["battery"],
            "epsilon": current_eps,
        })

        # Periodic logging to console
        if ep % 1000 == 0 or ep == episodes:
            recent = records[-500:]
            avg_r = np.mean([r["reward"] for r in recent])
            avg_d = np.mean([r["deliveries"] for r in recent])
            succ_r = np.mean([r["success"] for r in recent]) * 100.0
            avg_st = np.mean([r["steps"] for r in recent])
            elapsed = time.time() - start_time
            print(
                f"Ep {ep:5d}/{episodes} | Time: {elapsed:5.1f}s | "
                f"Eps: {current_eps:.3f} | Avg R: {avg_r:6.1f} | "
                f"Deliveries: {avg_d:4.2f}/3 | 3-Pack Succ: {succ_r:5.1f}% | "
                f"Steps: {avg_st:4.1f}"
            )

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time:.2f} seconds.")

    # 4. Save results
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # Save DataFrame
    df = pd.DataFrame(records)
    csv_path = f"results/training_log_{algo}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved training log to: {csv_path}")

    # Save Q-table
    model_path = f"results/q_table_{algo}.npy"
    agent.save(model_path)
    print(f"Saved trained Q-table to: {model_path}")

    # 5. Generate and save learning curves
    plot_learning_curve(df, algo)

    return agent, df


def plot_learning_curve(df: pd.DataFrame, algo: str):
    """
    Plots and saves dual-panel learning curves showing moving averages of
    return, success rate, and deliveries over episodes.
    """
    window = min(500, max(50, len(df) // 20))
    df["rolling_reward"] = df["reward"].rolling(window, min_periods=10).mean()
    df["rolling_success"] = (df["success"].rolling(window, min_periods=10).mean()) * 100.0
    df["rolling_deliveries"] = df["deliveries"].rolling(window, min_periods=10).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel 1: Episodic Return
    ax1.plot(df["episode"], df["reward"], color="lightgray", alpha=0.4, label="Raw Return")
    ax1.plot(df["episode"], df["rolling_reward"], color="#1f77b4", linewidth=2.0, label=f"Return (MA {window})")
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("Episodic Return (Reward)")
    ax1.set_title(f"Learning Performance: {algo.upper()} on Drone Delivery", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Dual axis for epsilon
    ax1_eps = ax1.twinx()
    ax1_eps.plot(df["episode"], df["epsilon"], color="#ff7f0e", linestyle="--", label="Epsilon (Exploration)")
    ax1_eps.set_ylabel("Epsilon", color="#ff7f0e")
    ax1_eps.tick_params(axis="y", labelcolor="#ff7f0e")
    ax1_eps.set_ylim(-0.05, 1.05)

    # Panel 2: Success Rate and Deliveries
    ax2.plot(df["episode"], df["rolling_success"], color="#2ca02c", linewidth=2.0, label=f"3-Package Success % (MA {window})")
    ax2.set_ylabel("Success Rate (%)", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(-5, 105)
    ax2.set_xlabel("Episode")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Dual axis for delivery count
    ax2_deliv = ax2.twinx()
    ax2_deliv.plot(df["episode"], df["rolling_deliveries"], color="#9467bd", linestyle="-.", linewidth=1.8, label=f"Avg Deliveries (MA {window})")
    ax2_deliv.set_ylabel("Deliveries Completed (out of 3)", color="#9467bd")
    ax2_deliv.tick_params(axis="y", labelcolor="#9467bd")
    ax2_deliv.set_ylim(-0.1, 3.2)

    # Combine legends for panel 2
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_2b, labels_2b = ax2_deliv.get_legend_handles_labels()
    ax2.legend(lines_2 + lines_2b, labels_2 + labels_2b, loc="lower right")

    plt.tight_layout()
    plot_path = f"results/plots/learning_curve_{algo}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved learning curve plot to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Q-Learning or SARSA agent on DroneDeliveryEnv")
    parser.add_argument("--algo", type=str, default="qlearning", choices=["qlearning", "sarsa"], help="Algorithm to train")
    parser.add_argument("--episodes", type=int, default=config.DEFAULT_EPISODES, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    train(algo=args.algo, episodes=args.episodes, seed=args.seed)
