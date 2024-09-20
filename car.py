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
    def __init__(self, max_speed, max_acceleration, max_steering_angle, mass=750, air_density=1.225, tire_grip=1.5):
        # Physical properties
        self.mass = mass  # kg (typical F1 car mass)
        self.air_density = air_density  # kg/m^3

        # Car state
        self.position = np.array([0.0, 0.0])  # x, y
        self.speed = 0.0  # m/s
        self.heading = 0.0  # radians

        # Control inputs
        self.acceleration = 0.0  # m/s^2
        self.brake_force = 0.0  # Braking force (applied when braking)
        self.steering_angle = 0.0  # radians

        # Maximum limits
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_steering_angle = max_steering_angle

        # Aerodynamic coefficients
        self.drag_coefficient = 0.9  # Typical for an F1 car
        self.frontal_area = 1.5  # m^2, approximate frontal area

        # Downforce (helps with cornering stability)
        self.downforce_coefficient = 3.0  # A typical value for a high-performance car

        # Tire properties
        self.tire_grip = tire_grip  # Coefficient of friction
        self.wheel_base = 1.6  # m, distance between front and rear axles
        self.tire_wear = 1.0  # No tire wear at start (1.0 is optimal grip)

        # Cornering properties
        self.weight_transfer = 0.05  # Percentage of weight transferred during cornering
        self.lateral_grip = tire_grip  # Lateral grip (slip)

    def update(self, throttle_input, brake_input, steering_input, dt):
        # Map inputs to physical values
        self.acceleration = throttle_input * self.max_acceleration
        self.steering_angle = steering_input * self.max_steering_angle

        # Apply braking force
        self.brake_force = brake_input * self.max_acceleration * 2.0  # Brakes are typically more powerful than acceleration
        self.acceleration -= self.brake_force

        # Aerodynamic drag force: F_drag = 0.5 * rho * Cd * A * v^2
        drag_force = 0.5 * self.air_density * self.drag_coefficient * self.frontal_area * self.speed ** 2
        drag_acceleration = drag_force / self.mass

        # Downforce increases tire grip based on speed: F_downforce = Cd_downforce * v^2
        downforce = 0.5 * self.air_density * self.downforce_coefficient * self.speed ** 2
        effective_grip = self.tire_grip * (1.0 + downforce / self.mass) * self.tire_wear

        # Update speed with acceleration, braking, and drag
        self.speed += (self.acceleration - drag_acceleration) * dt
        self.speed = np.clip(self.speed, 0.0, self.max_speed)

        # Calculate slip angle (beta)
        if self.speed != 0:
            beta = np.arctan((self.wheel_base / 2) * np.tan(self.steering_angle) / self.wheel_base)
        else:
            beta = 0.0

        # Lateral grip: reduces car speed based on steering angle
        lateral_force = self.speed * np.tan(self.steering_angle) * self.weight_transfer
        lateral_grip_limit = effective_grip * self.mass * 9.81  # Maximum cornering force
        if lateral_force > lateral_grip_limit:
            self.speed *= 0.95  # Reduce speed if lateral grip is exceeded (simulating slipping)
        
        # Update heading
        self.heading += (self.speed / self.wheel_base) * np.sin(beta) * dt
        self.heading = wrap_angle(self.heading)

        # Update position
        dx = self.speed * np.cos(self.heading) * dt
        dy = self.speed * np.sin(self.heading) * dt
        self.position += np.array([dx, dy])

        # Tire wear model: tires degrade over time, affecting grip
        self.tire_wear -= 0.0001 * np.abs(self.speed) * dt  # Arbitrary wear rate
        self.tire_wear = np.clip(self.tire_wear, 0.5, 1.0)  # Minimum tire wear

    def reset(self, position, heading):
        self.position = np.array(position)
        self.heading = heading
        self.speed = 0.0
        self.acceleration = 0.0
        self.steering_angle = 0.0
        self.tire_wear = 1.0  # Reset tire wear to full

# Example usage:
car = Car(max_speed=300, max_acceleration=15, max_steering_angle=np.radians(30))
print(f"Car initialized at position {car.position} with heading {car.heading}.")
