#!/usr/bin/env python3
#---------------------------------
# AASMA - Environment
# File : env.py
#
# @ start date          22 04 2020
# @ last update         25 04 2020
#---------------------------------
# TODO : Add character behaviour
# TODO: Spawn a new window with the current time step and other metrics

#---------------------------------
# Imports
#---------------------------------
import os
import re, json
import numpy as np

import pygame
import tkinter as tk

from aasma.env.about_window import Application

#---------------------------------
# Constants
#---------------------------------
CONFIG_FIELDS = {
    # Dimensionality
    'grid_dim': 'dim',
    'cell_size': 'int',
    # Spawn
    'mountain': 'prob',
    'green': 'prob',
    'yellow': 'prob',
    'red': 'prob',
    # Evolution
    'green-yellow': 'ts',
    'yellow-red': 'ts',
    'red-fire': 'ts',
    'fire-green': 'ts'
}

HEATMAP_COLORS = {
    'green': (0, 250, 0),
    'yellow': (255, 204, 0),
    'red': (250, 0, 0),
    'mountain': (153, 102, 51), # brown
    'fire': (0, 0, 0)
}

HEATMAP_TRANSITION_GUIDE = ('green', 'yellow', 'red', 'fire')

#---------------------------------
# Sprites
#---------------------------------
PATH = os.path.realpath(__file__)
# Remove filename
PATH = PATH[:len(PATH) - 6]

# Sprites themselves
MOUNTAIN_SPRITE_FILEPATH = os.path.join(PATH, '../../mountain.png')
FIRE_SPRITE_FILEPATH = os.path.join(PATH, '../../fire.png')

#CHARACTER_SPRITE_FILEPATH = os.path.join(PATH, '../../drone.png')

#---------------------------------
# class Environment
#---------------------------------
class Environment:
    def __init__(self, config_file):
        self.__config = self.__parse_config(config_file)

        self.__matrix_repr = []

        # Build the grid world and spawn objects
        self.__build()

    def __parse_config(self, filename):
        if not re.search('.*\.json', filename):
            raise ValueError('Config file must be a .json')

        try:
            with open(filename, 'r') as config_file:
                content = json.load(config_file)
        except:
            raise ValueError('Config file IO failure')

        # Error handle the json content
        for field in CONFIG_FIELDS:
            if field not in content:
                raise ValueError('Missing config field \'{}\''.format(field))
            elif not self.__config_check(field, content[field]):
                raise ValueError(
                    'Invalid config value for \'{}\''.format(field)
                )

        # Obtain the game window dimensions
        dimensions = content['grid_dim'].split(',')
        self.__WINDOW_WIDTH = int(dimensions[0])
        self.__WINDOW_HEIGHT = int(dimensions[1])

        # Also check the correctness of the law of total probability
        total_prob = float(content['green']) + \
            float(content['yellow']) + float(content['red']) + \
            float(content['mountain'])

        if round(total_prob, 2) != 1:
            print("Probability deducted: {}".format(total_prob))
            raise ValueError(
                'Violated law of total probability. Change config file'
            )

        # Check compatibility of screen dimensions and cell size
        cell_size = int(content['cell_size'])

        if not (self.__WINDOW_WIDTH % cell_size == 0 and \
            self.__WINDOW_HEIGHT % cell_size == 0):
            raise ValueError('\'cell_size\' and \'grid_dim\' are incompatible')

        return content

    def __config_check(self, field, value):
        data_type = CONFIG_FIELDS[field]
        if data_type == 'prob':
            try:
                return 0 <= float(value) <= 1
            except:
                return False
        elif data_type in ('ts', 'int'):
            try:
                return int(value) > 1
            except:
                return False
        elif data_type == 'dim':
            try:
                w, h = value.split(',')
                return int(w) > 0 and int(h) > 0
            except:
                return False
        else:
            raise ValueError('Unknown data_type of \'{}\''.format(field))

    def __build(self):
        # Start the environment engine
        pygame.init()
        pygame.display.set_caption('AASMA Environment')
        self.__screen = pygame.display.set_mode(
            (self.__WINDOW_WIDTH, self.__WINDOW_HEIGHT)
        )

        # Draw the environment itself
        self.__draw_and_spawn()

        # Start the environment engine clock
        self.__clock = pygame.time.Clock()

    def __draw_and_spawn(self):
        # Fill the window
        # This will eventually be (after heatmap placement)
        # the lines which separate cells (thickness=1)
        self.__screen.fill((0, 0, 0))

        # Init the heat map
        # Use the heatmap heat signature probabilities
        # provided in config.json

        # Heat signature probability distribution
        p_distribution = np.array([
            float(self.__config['green']),
            float(self.__config['yellow']),
            float(self.__config['red']),
            float(self.__config['mountain'])
        ])

        # Populate each cell with a heat signature and biome elements
        cell_size = int(self.__config['cell_size'])
        for x in range(0, self.__WINDOW_WIDTH, cell_size):
            row_repr = []
            for y in range(0, self.__WINDOW_HEIGHT, cell_size):
                color = np.random.choice(
                    ('green', 'yellow', 'red', 'mountain'), 1,
                    p=p_distribution
                )[0]

                cell = pygame.Rect(
                    x + 2, y + 2,
                    cell_size - 2, cell_size - 2
                )

                # Just load mountains according to the probability
                # given in config.json
                p_mountain = float(self.__config['mountain'])

                pygame.draw.rect(
                    self.__screen, HEATMAP_COLORS[color],
                    cell, 0 # 0 = fill cell
                )

                # Add cell to environment matrix representation
                # cell = [color, time step]
                row_repr.append([color, 0])

                if color == 'mountain':
                    mountain = pygame.image.load(
                        MOUNTAIN_SPRITE_FILEPATH
                    )
                    mountain = pygame.transform.scale(
                        mountain, (cell_size - 2, cell_size - 2)
                    )
                    self.__screen.blit(mountain, (x + 2, y + 2))

            # Add row to the matrix representation of the environment
            self.__matrix_repr.append(row_repr)

        pygame.display.flip()

        root = tk.Tk()
        app = Application(master=root)
        app.mainloop()

    def run(self):
        EXECUTE = True
        while EXECUTE:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        EXECUTE = False

            # Run an update on the env grid
            # (without external interference)
            cell_size = int(self.__config['cell_size'])
            for x in range(0, self.__WINDOW_WIDTH, cell_size):
                for y in range(0, self.__WINDOW_HEIGHT, cell_size):
                    # Update matrix representation of the environment
                    cell = self.__matrix_repr[x // cell_size][y // cell_size]
                    color, ts = cell

                    # Next heatmap signature to be displayed
                    if color != 'mountain':
                        i = HEATMAP_TRANSITION_GUIDE.index(color)
                        next_hm_signature = HEATMAP_TRANSITION_GUIDE[(i + 1) % len(HEATMAP_TRANSITION_GUIDE)]

                        if ts >= self.__config[color + '-' + next_hm_signature]:
                            self.__matrix_repr[x // cell_size][y // cell_size][0] = next_hm_signature
                            self.__matrix_repr[x // cell_size][y // cell_size][1] = 0
                        else:
                            self.__matrix_repr[x // cell_size][y // cell_size][1] += 1

                        # Update pygame environment
                        cell = pygame.Rect(
                            x + 2, y + 2,
                            cell_size - 2, cell_size - 2
                        )

                        pygame.draw.rect(self.__screen, HEATMAP_COLORS[self.__matrix_repr[x // cell_size][y // cell_size][0]], cell, 0)

                        #Adds a fire sprite to the fire blocks
                        if self.__matrix_repr[x // cell_size][y // cell_size][0] == 'fire':
                            fire = pygame.image.load(FIRE_SPRITE_FILEPATH)
                            fire = pygame.transform.scale(fire, (cell_size - 2, cell_size - 2))
                            self.__screen.blit(fire, (x + 2, y + 2))

            pygame.display.flip()
            # Handle external events
            # TODO

            # Tick the clock with 15Hz (15 frame per second)
            self.__clock.tick(1)
