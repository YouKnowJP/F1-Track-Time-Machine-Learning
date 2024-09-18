#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:37:35 2024

@author: youknowjp
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from pygame.locals import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time  # Added for controlling render speed

# Utility functions
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Car class with simplified dynamics
class Car:
    def __init__(self, max_speed, max_acceleration, max_steering_angle, mass=750, air_density=1.225):
        # Physical properties
        self.mass = mass  # kg (typical F1 car mass)
        self.air_density = air_density  # kg/m^3

        # Car state
        self.position = np.array([0.0, 0.0])  # x, y
        self.speed = 0.0  # m/s
        self.heading = 0.0  # radians

        # Control inputs
        self.acceleration = 0.0  # m/s^2
        self.steering_angle = 0.0  # radians

        # Maximum limits
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_steering_angle = max_steering_angle

        # Aerodynamic coefficients
        self.drag_coefficient = 0.9  # Typical for an F1 car
        self.frontal_area = 1.5  # m^2, approximate frontal area

        # Tire properties
        self.tire_grip = 1.5  # Coefficient of friction
        self.wheel_base = 1.6  # m, distance between front and rear axles

    def update(self, throttle_input, steering_input, dt):
        # Map inputs to physical values
        self.acceleration = throttle_input * self.max_acceleration
        self.steering_angle = steering_input * self.max_steering_angle

        # Aerodynamic drag force: F_drag = 0.5 * rho * Cd * A * v^2
        drag_force = 0.5 * self.air_density * self.drag_coefficient * self.frontal_area * self.speed ** 2
        drag_acceleration = drag_force / self.mass

        # Update speed with acceleration and drag
        self.speed += (self.acceleration - drag_acceleration) * dt
        self.speed = np.clip(self.speed, 0.0, self.max_speed)

        # Calculate slip angle (beta)
        if self.speed != 0:
            beta = np.arctan((self.wheel_base / 2) * np.tan(self.steering_angle) / self.wheel_base)
        else:
            beta = 0.0

        # Update heading
        self.heading += (self.speed / self.wheel_base) * np.sin(beta) * dt
        self.heading = wrap_angle(self.heading)

        # Update position
        dx = self.speed * np.cos(self.heading) * dt
        dy = self.speed * np.sin(self.heading) * dt
        self.position += np.array([dx, dy])

    def reset(self, position, heading):
        self.position = np.array(position)
        self.heading = heading
        self.speed = 0.0
        self.acceleration = 0.0
        self.steering_angle = 0.0

# Simplified Track class
class Track:
    def __init__(self, name='circular', friction=1.0):
        self.name = name
        self.friction = friction
        self.waypoints = self.create_track()
        self.length = len(self.waypoints)
        # For this simplified version, we will not consider elevations or varying friction

    def create_track(self):
        num_points = 100  # Reduced number of waypoints
        if self.name == 'circular':
            radius = 500  # meters
            angles = np.linspace(0, 2 * np.pi, num_points)
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            waypoints = np.vstack((x, y)).T
        else:
            # Default to circular track
            waypoints = self.create_track()
        return waypoints

# Simplified Logger class (logging disabled to speed up)
class Logger:
    def __init__(self, filename='training_log.csv'):
        self.filename = filename
        self.fields = ['step', 'position_x', 'position_y', 'speed', 'heading', 'action_steering', 'action_throttle', 'reward']
        # Disabled logging for speed
        # with open(self.filename, 'w', newline='') as f:
        #     writer = csv.DictWriter(f, fieldnames=self.fields)
        #     writer.writeheader()

    def log(self, step, car_state, action, reward):
        pass  # Disabled logging for speed

# The main environment class
class F1TrackEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(F1TrackEnv, self).__init__()
        # Initialize track
        self.track = Track(name='circular')  # Use simpler 'circular' track
        # Initialize car
        self.car = Car(max_speed=100.0, max_acceleration=20.0, max_steering_angle=np.deg2rad(30))
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-1000.0, -1000.0, 0.0, -np.pi]),
            high=np.array([1000.0, 1000.0, self.car.max_speed, np.pi]),
            dtype=np.float32
        )
        # Time step
        self.dt = 0.1  # seconds
        # Logger
        self.logger = Logger()
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Reset car and environment state
        start_pos = self.track.waypoints[0]
        self.car.reset(start_pos, 0.0)
        self.current_waypoint = 0
        self.current_step = 0
        observation = np.array([
            self.car.position[0],
            self.car.position[1],
            self.car.speed,
            self.car.heading
        ], dtype=np.float32)
        info = {}
        # Initialize rendering
        self.screen = None
        return observation, info

    def step(self, action):
        # Unpack action
        steering_input = np.clip(action[0], -1.0, 1.0)
        throttle_input = np.clip(action[1], -1.0, 1.0)

        # Update car state
        self.car.update(throttle_input, steering_input, self.dt)

        # Update waypoint index
        distance_to_waypoint = np.linalg.norm(self.track.waypoints[self.current_waypoint] - self.car.position)
        if distance_to_waypoint < 10.0 and self.current_waypoint < self.track.length - 1:
            self.current_waypoint += 1

        # Calculate reward (Simplified)
        deviation = np.linalg.norm(self.track.waypoints[self.current_waypoint] - self.car.position)
        reward = -deviation  # Encourage minimizing deviation from the track

        # Off-track penalty
        off_track = deviation > 20.0
        if off_track:
            terminated = True
            reward -= 1000  # Large penalty for going off-track
        else:
            terminated = False

        # Check if race is completed
        if self.current_waypoint >= self.track.length - 1:
            terminated = True
            reward += 1000  # Bonus for completing the lap

        truncated = False  # No time limit truncation

        # Construct observation
        observation = np.array([
            self.car.position[0],
            self.car.position[1],
            self.car.speed,
            self.car.heading
        ], dtype=np.float32)

        info = {}

        # Disabled logging for speed
        # self.logger.log(self.current_step, self.car, action, reward)
        self.current_step += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        if not hasattr(self, 'screen') or self.screen is None:
            pygame.init()
            self.screen_size = (800, 800)
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption('F1 Simulation')
            self.clock = pygame.time.Clock()

            # Transform track coordinates to fit the screen
            self.track_surface = pygame.Surface(self.screen_size)
            self.track_surface.fill((0, 0, 0))
            transformed_track = self.transform_track(self.track.waypoints)
            pygame.draw.lines(self.track_surface, (255, 255, 255), True, transformed_track, 2)

        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                self.close()
                pygame.quit()
                quit()

        # Clear screen
        self.screen.blit(self.track_surface, (0, 0))

        # Draw the car
        car_pos = self.transform_point(self.car.position)
        car_heading = -np.degrees(self.car.heading)  # Pygame uses negative degrees for rotation

        car_image = pygame.Surface((20, 10), pygame.SRCALPHA)
        pygame.draw.polygon(car_image, (255, 0, 0), [(0, 5), (20, 0), (20, 10)])
        rotated_image = pygame.transform.rotate(car_image, car_heading)
        rect = rotated_image.get_rect(center=car_pos)
        self.screen.blit(rotated_image, rect.topleft)

        # Display speed
        font = pygame.font.SysFont(None, 24)
        speed_text = font.render(f'Speed: {self.car.speed:.2f} m/s', True, (255, 255, 255))
        self.screen.blit(speed_text, (10, 10))

        # Update display
        pygame.display.flip()
        self.clock.tick(60)  # Limit to 60 FPS

    def transform_track(self, track_points):
        # Transform track coordinates to fit screen size
        scale = 0.6  # Adjust as needed
        offset = np.array(self.screen_size) / 2
        transformed = []
        for point in track_points:
            x, y = point * scale + offset
            transformed.append((int(x), int(y)))
        return transformed

    def transform_point(self, point):
        scale = 0.6
        offset = np.array(self.screen_size) / 2
        x, y = point * scale + offset
        return int(x), int(y)

    def close(self):
        if hasattr(self, 'screen') and self.screen is not None:
            pygame.quit()
            self.screen = None

# Function to run the agent and record performance data
def run_agent(env, model_path, render=False):
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

# Main code to run the environment
if __name__ == "__main__":
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
