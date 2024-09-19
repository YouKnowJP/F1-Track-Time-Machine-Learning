#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:28:49 2024

@author: youknowjp
"""

# env/f1_track_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from pygame.locals import *
from stable_baselines3.common.env_checker import check_env

from .car import Car
from .track import Track
from .logger import Logger
from utils.math_utils import wrap_angle

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
