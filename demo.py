"""
demo.py
-------
Step-by-step live demonstration of a trained RL policy on DroneDeliveryEnv.

Matches the required Mini-Project live demo sequence:
  Step 1: Start environment / reset state
  Step 2: Show current observation / state (readable breakdown)
  Step 3: Show available actions and selected action (greedy vs exploratory)
  Step 4: Execute action and show environment response
  Step 5: Show reward and next state
  Step 6: Show Q-value update step (old Q, target, new Q) live
  Step 7: Repeat until episode termination
  Step 8: Display final summary, episode outcome, and performance metrics

Usage:
    python demo.py                     # Interactive mode (press Enter to step)
    python demo.py --fast              # Automated playback without pausing
    python demo.py --algo sarsa        # Demonstrate SARSA agent
    python demo.py --seed 123          # Run with a specific scenario seed
"""

import argparse
import os
import sys
import time
import numpy as np

import config
from drone_env import DroneDeliveryEnv, decode_state
from q_agent import QLearningAgent
from sarsa_agent import SARSAAgent


def run_demo(algo: str = "qlearning", fast: bool = False, delay: float = 0.2, seed: int = config.SEED):
    """
    Executes an interactive or automated step-by-step demonstration episode.
    """
    q_table_path = f"results/q_table_{algo}.npy"
    if not os.path.exists(q_table_path):
        print(f"Error: Trained Q-table '{q_table_path}' not found!")
        print(f"Please run 'python train.py --algo {algo}' first to generate the trained model.")
        sys.exit(1)

    print("\n" + "=" * 75)
    print(f"       LIVE RL DEMONSTRATION: {algo.upper()} AGENT ON DRONE DELIVERY")
    print("=" * 75)
    print(f"Algorithm         : {algo.upper()}")
    print(f"Loaded Q-Table    : {q_table_path}")
    print(f"Execution Mode    : {'Fast (Automated)' if fast else 'Interactive (Press Enter to Step)'}")
    print(f"Scenario Seed     : {seed}")
    print("=" * 75 + "\n")

    # 1. Initialize environment and load agent
    env = DroneDeliveryEnv(render_mode="human")
    env.seed(seed)

    if algo == "qlearning":
        agent = QLearningAgent(num_states=env.num_states, num_actions=config.NUM_ACTIONS)
    else:
        agent = SARSAAgent(num_states=env.num_states, num_actions=config.NUM_ACTIONS)

    agent.load(q_table_path)
    # Set to exploitation / evaluation mode (or very small exploration)
    agent.epsilon = 0.0

    # Step 1: Start environment
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    step = 0
    done = False

    print("Step 1: Environment initialized.")
    env.render()

    if not fast:
        input("\nPress [Enter] to start step-by-step execution...")

    # Step 2-7: Execution loop
    while not done:
        step += 1
        print("\n" + "-" * 75)
        print(f">>> STEP {step} <<<")
        print("-" * 75)

        # Step 2: Show current state
        x, y, bat, car, d_mask, ub = decode_state(obs)
        print(f"[State {obs}]")
        print(f"  • Position        : ({x}, {y})")
        print(f"  • Battery         : {bat}/{config.MAX_BATTERY} {'[LOW]' if bat <= 5 else '[OK]'}")
        print(f"  • Carrying Payload: {'YES (Loaded)' if car else 'NO (Empty)'}")
        print(f"  • Deliveries Done : {[config.DROPOFF_NAMES[i] for i in range(3) if (d_mask & (1 << i))] or ['None']}")
        print(f"  • Urgent Bucket   : {['0: Safe', '1: Warning', '2: Expired'][ub]}")

        # Step 3: Show available actions & Q-values
        q_values = agent.q_table[obs]
        print("\n[Available Actions & Current Q-Values]:")
        for a_id in range(config.NUM_ACTIONS):
            name = config.ACTION_NAMES[a_id]
            is_best = " (BEST)" if a_id == np.argmax(q_values) else ""
            print(f"  Action {a_id} [{name:<15}]: Q = {q_values[a_id]:7.2f}{is_best}")

        action, is_greedy = agent.select_action(obs, greedy=True)
        act_type = "GREEDY (Exploitative)" if is_greedy else "EXPLORATORY (Random)"
        print(f"\n[Decision]: Chosen Action = {action} ({config.ACTION_NAMES[action]}) via {act_type}")

        # Step 4 & 5: Environment response & reward
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        print(f"\n[Environment Transition]:")
        print(f"  • Step Reward     : {reward:+6.1f} (Running Return: {total_reward:+6.1f})")
        print(f"  • New Position    : ({next_info['x']}, {next_info['y']})")
        print(f"  • New Battery     : {next_info['battery']}/{config.MAX_BATTERY}")
        print(f"  • Status          : {'TERMINATED' if terminated else ('TRUNCATED' if truncated else 'IN PROGRESS')}")

        # Step 6: Learning / Bellman Update step demonstration
        old_q = float(agent.q_table[obs, action])
        if algo == "qlearning":
            best_next = float(np.max(agent.q_table[next_obs])) if not terminated else 0.0
            target = float(reward + config.GAMMA * best_next)
            formula_desc = f"r + gamma * max_a' Q(s', a') = {reward:.1f} + {config.GAMMA} * {best_next:.2f} = {target:.2f}"
        else:
            next_action, _ = agent.select_action(next_obs, greedy=True)
            next_q = float(agent.q_table[next_obs, next_action]) if not terminated else 0.0
            target = float(reward + config.GAMMA * next_q)
            formula_desc = f"r + gamma * Q(s', a') = {reward:.1f} + {config.GAMMA} * {next_q:.2f} = {target:.2f}"

        td_error = target - old_q
        simulated_new_q = old_q + config.ALPHA * td_error

        print(f"\n[Bellman Temporal-Difference Update Check]:")
        print(f"  • Target Formula  : {formula_desc}")
        print(f"  • Old Q(s, a)     : {old_q:.3f}")
        print(f"  • TD Target       : {target:.3f}")
        print(f"  • TD Error (delta): {td_error:.3f}")
        print(f"  • Updated Q(s, a) : {simulated_new_q:.3f}")

        # Render visual grid
        print("\n[Grid State]:")
        env.render()

        obs = next_obs

        # Step control
        if not fast and not done:
            input("\nPress [Enter] to proceed to next step...")
        elif fast:
            time.sleep(delay)

    # Step 8: Final outcome summary
    print("\n" + "=" * 75)
    print("                     DEMONSTRATION COMPLETED")
    print("=" * 75)
    outcome = "SUCCESS (All 3 packages delivered!)" if next_info["all_delivered"] else (
        "FAILED (Battery depleted)" if next_info["battery"] == 0 else "TRUNCATED (Max steps reached)"
    )
    print(f"Final Outcome         : {outcome}")
    print(f"Total Steps Taken     : {step}")
    print(f"Final Return (Reward) : {total_reward:.2f}")
    print(f"Final Battery Level   : {next_info['battery']}/{config.MAX_BATTERY}")
    print(f"Deliveries Completed  : {next_info['delivered_count']} / 3")
    print(f"Urgent Package (D3)   : {'DELIVERED ON TIME' if next_info['urgent_met_on_time'] else 'LATE OR MISSED'}")
    plot_file = f"results/plots/learning_curve_{algo}.png"
    print(f"Associated Learning Curve: {plot_file}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Step-by-Step RL Policy Demonstration")
    parser.add_argument("--algo", type=str, default="qlearning", choices=["qlearning", "sarsa"], help="Algorithm Q-table to demonstrate")
    parser.add_argument("--fast", action="store_true", help="Run automatically without waiting for Enter key")
    parser.add_argument("--delay", type=float, default=0.15, help="Delay in seconds between steps when running with --fast")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Random seed for the episode")
    args = parser.parse_args()

    run_demo(algo=args.algo, fast=args.fast, delay=args.delay, seed=args.seed)
