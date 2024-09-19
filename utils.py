#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:27:37 2024

@author: youknowjp
"""

# utils/math_utils.py

import numpy as np

def wrap_angle(angle):
    """Wrap angle between -π and π."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
