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
