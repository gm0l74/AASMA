#!/usr/bin/env python3
#---------------------------------
# AASMA - Environment
# File : Mountain.py
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
class Mountain(pygame.sprite.Sprite):
    def __init__(self, color, x, y):
        self.__init__()

        self.image = pygame.image.load("mountain.png").convert()
        self.image.set_colorkey(color)

        self.rect = self.image.get_rect()

        # TODO
        # return a width and height of an image
        self.size = self.image.get_size()

        # create a 2x bigger image than self.image
        self.bigger_img = pygame.transform.scale(self.image, (int(self.size[0]*2), int(self.size[1]*2)))
        # draw bigger image to screen at x=100 y=100 position
        self.screen.blit(self.bigger_img, [100,100])
