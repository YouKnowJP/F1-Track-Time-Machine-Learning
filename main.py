#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:29:53 2024

@author: youknowjp
"""

# main.py

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env.f1_track_env import F1TrackEnv
from agents.run_agent import run_agent

def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Initialize environment and check it
    env = F1TrackEnv()
    check_env(env)

    # Instantiate the agent
    policy_kwargs = dict(
        net_arch=dict(pi=[32, 32], vf=[32, 32])  # Smaller network with two hidden layers
    )
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

    # Training parameters (Reduced for faster execution)
    total_timesteps = 10000  # Further reduced total timesteps
    save_interval = 2000     # Adjusted save interval
    num_iterations = total_timesteps // save_interval  # This will be 5 iterations

    # Lists to store metrics
    total_rewards = []
    timesteps = []

    for i in range(1, num_iterations + 1):
        model.learn(total_timesteps=save_interval, reset_num_timesteps=False)
        model.save(f"ppo_f1_model_{i * save_interval}")
        print(f"Saved model at {i * save_interval} timesteps")

        # Run the agent and collect total reward
        # Set render=True to visualize the car running
        positions, speeds, headings, total_reward = run_agent(env, f"ppo_f1_model_{i * save_interval}", render=True)
        total_rewards.append(total_reward)
        timesteps.append(i * save_interval)

    # Plot total reward over time
    plt.figure()
    plt.plot(timesteps, total_rewards, marker='o')
    plt.xlabel('Timesteps')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Time')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
