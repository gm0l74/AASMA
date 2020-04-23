#!/usr/bin/env python3
#---------------------------------
# AASMA - Environment
# File : Fire.py
#
# @ start date          22 04 2020
# @ last update         23 04 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import pygame

#---------------------------------
# class Fire
#---------------------------------
class Fire(pygame.sprite.Sprite):
    def __init__(self, color, x, y):
        super.__init__()

        # Load the fire sprite
        self.image = pygame.image.load("fire.png").convert()
        self.image.set_colorkey(color)

        # TODO
