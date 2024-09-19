#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:28:18 2024

@author: youknowjp
"""

# env/car.py

import numpy as np
from utils.math_utils import wrap_angle

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
