#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:29:37 2024

@author: youknowjp
"""

# rendering/renderer.py

import pygame
from pygame.locals import *

class Renderer:
    def __init__(self, screen_size=(800, 800), title='F1 Simulation'):
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.screen_size = screen_size

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.close()
                pygame.quit()
                quit()

    def close(self):
        pygame.quit()
