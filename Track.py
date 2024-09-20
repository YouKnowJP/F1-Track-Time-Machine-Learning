#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:28:26 2024

@author: youknowjp
"""

# env/track.py

import numpy as np

class Track:
    def __init__(self, name='circular', friction=1.0):
        self.name = name
        self.friction = friction
        self.waypoints = self.create_track()
        self.length = len(self.waypoints)

    def create_track(self):
        num_points = 500  # Increase number of waypoints for more detail
        if self.name == 'circular':
            return self.create_circular_track(num_points)
        elif self.name == 'complex':
            return self.create_complex_track(num_points)
        else:
            return self.create_circular_track(num_points)  # Default to circular

    def create_circular_track(self, num_points):
        radius = 500  # meters
        angles = np.linspace(0, 2 * np.pi, num_points)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        waypoints = np.vstack((x, y)).T
        return waypoints

    def create_complex_track(self, num_points):
        # Create a track with straight sections, sharp turns, and chicanes
        waypoints = []
        
        # Add a long straight section
        straight_length = 300
        x = np.linspace(0, straight_length, num_points // 10)
        y = np.zeros(num_points // 10)
        waypoints.extend(zip(x, y))
        
        # Add a sharp turn (90 degrees right)
        radius = 50
        angles = np.linspace(0, np.pi / 2, num_points // 20)
        x = straight_length + radius * np.sin(angles)
        y = radius * (1 - np.cos(angles))
        waypoints.extend(zip(x, y))
        
        # Add a chicane (S-curve)
        chicane_length = 150
        chicane_x = np.linspace(x[-1], x[-1] + chicane_length, num_points // 10)
        chicane_y = 10 * np.sin(2 * np.pi * (chicane_x - x[-1]) / chicane_length)
        waypoints.extend(zip(chicane_x, chicane_y + y[-1]))
        
        # Add a wide turn (180 degrees left)
        radius = 200
        angles = np.linspace(np.pi / 2, 3 * np.pi / 2, num_points // 15)
        x = chicane_x[-1] + radius * np.sin(angles)
        y = chicane_y[-1] + radius * (1 - np.cos(angles))
        waypoints.extend(zip(x, y))
        
        # Add another straight section
        straight_length = 400
        x = np.linspace(x[-1], x[-1] + straight_length, num_points // 8)
        y = np.full(num_points // 8, y[-1])
        waypoints.extend(zip(x, y))
        
        # Convert waypoints list to numpy array
        waypoints = np.array(waypoints)
        return waypoints

# Example usage:
track = Track(name='complex')
print(f"Track with {track.length} waypoints created.")
