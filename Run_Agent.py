#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:29:22 2024

@author: youknowjp
"""

# agents/run_agent.py

import time
import numpy as np
from stable_baselines3 import PPO

def run_agent(env, model_path, render=False):
    """
    Load a trained PPO model and run it in the given environment.

    Args:
        env: The Gym environment.
        model_path (str): Path to the saved PPO model.
        render (bool): Whether to render the environment.

    Returns:
        positions (list): List of car positions.
        speeds (list): List of car speeds.
        headings (list): List of car headings.
        total_reward (float): Total accumulated reward.
    """
    # Load the model without passing the environment
    model = PPO.load(model_path)
    obs, info = env.reset()
    positions = []
    speeds = []
    headings = []
    total_reward = 0

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(env.car.position.copy())
        speeds.append(env.car.speed)
        headings.append(env.car.heading)
        total_reward += reward
        if render:
            env.render()
            # Optional: Add delay if rendering is too fast
            time.sleep(0.05)
        if terminated or truncated:
            break

    env.close()
    return positions, speeds, headings, total_reward
