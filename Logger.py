#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:28:38 2024

@author: youknowjp
"""

# env/logger.py

import csv

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
