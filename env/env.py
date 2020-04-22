#!/usr/bin/env python3
#---------------------------------
# AASMA - Environment
# File : env.py
#
# @ start date          22 04 2020
# @ last update         22 04 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import os
import re, json
import pygame
import numpy as np

from aasma.env.sprites.Mountain import Mountain

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
    'red-fire': 'ts'
}

HEATMAP_COLORS = {
    'green': (0, 250, 0),
    'yellow': (245, 200, 66),
    'red': (250, 0, 0)
}

#---------------------------------
# Sprites
#---------------------------------
PATH = os.path.realpath(__file__)
# Remove filename
PATH = ROOT_PATH[:len(PATH) - 6]

# Sprites themselves
MOUNTAIN = pg.image.load(
    os.path.join(PATH, './sprites/mountain.png')
).convert_alpha()

#---------------------------------
# class Environment
#---------------------------------
class Environment:
    def __init__(self, config_file):
        self.__config = self.__parse_config(config_file)

        # Build the grid world and spawn objects
        self.__build() ; self.__spawn()

    def __parse_config(self, filename):
        if not re.search('.*\.json', filename):
            raise ValueError('Config file must be a .json')

        try:
            content = json.load(filename)
        except:
            raise ValueError('File couldn\'t be found')

        # Error handle the json content
        for field in CONFIG_FIELDS:
            if field not in content:
                raise ValueError(f'Missing config field \'{field}\'')
            elif (e := self.__config_check(field, content[field])):
                print(f'\033[31m{e}\033[0m')
                raise ValueError(f'invalid config value for \'{field}\'')

        # Obtain the game window dimensions
        dimensions = content['grid_dim'].split(',')
        self.__WINDOW_WIDTH = int(dimensions[0])
        self.__WINDOW_HEIGHT = int(dimensions[1])

        # Also check the correctness of the law of total probability
        total_prob = float(content['green']) + \
            float(content['yellow']) + float(content['red'])

        if total_prob != 1:
            raise ValueError(f'violated law of total probability')

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
                w, h = int(w), int(h)
            except:
                raise ValueError('That dimension is unsupported')
        else:
            raise ValueError(f'Unknown data_type of \'{field}\'')

    def __build(self):
        # Start the environment engine
        pygame.init()
        self.__screen = pygame.display.set_mode(())

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
            float(self.__config['red'])
        ])

        cell_size = self.__config['cell_size']

        # Populate each cell with a heat signature and biome elements
        for x in range(0, self.__WINDOW_WIDTH, cell_size):
            for y in range(0, self.__WINDOW_HEIGHT, cell_size):
                color = np.random.choice(
                    HEATMAP_COLORS.keys(), 1,
                    p=p_distribution
                )

                cell = pygame.Rect(
                    x * block_size + 1, y * block_size + 1,
                    block_size - 1, block_size - 1
                )

                # Just load mountains according to the probability
                # given in config.json
                p_mountain = float(self.__config['mountain'])

                has_mountain = np.random.choice(
                    [True False], 1,
                    p=[p_mountain, 1 - p_mountain]
                )

                if has_mountain:
                    # Load a mountain into this cell
                    # TODO
                    Mountain(HEATMAP_COLORS[color], x, y)
                    mountain_sprite.rect = cell

                pygame.draw.rect(
                    self.__screen, HEATMAP_COLORS[color],
                    cell, 0 # 0 = fill cell
                )

    def run(self):
        EXECUTE = True
        while EXECUTE:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        EXECUTE = False

            # Handle external events
            # TODO

            # Handle changes in environment
            # TODO

            # Tick the clock with 1Hz (1 frame per second)
            clock.tick(1)
