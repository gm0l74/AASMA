#!/usr/bin/env python3
#---------------------------------
# AASMA - Environment
# File : env.py
#
# @ start date          22 04 2020
# @ last update         28 04 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import os
import re, json
import hashlib
from datetime import datetime
from random import randint

import numpy as np
import pygame

#---------------------------------
# PWD
#---------------------------------
PATH = os.path.realpath(__file__)
# Remove filename
PATH = PATH[:len(PATH) - 6]

#---------------------------------
# Constants
#---------------------------------
CONFIG_FIELDS = {
    # Dimensionality
    'grid_dim': 'int_tuple', # int_tuple is (int > 0, int > 0)
    'cell_size': 'int',
    # Spawn
    'mountain': 'prob', # prob belongs to [0, 1]
    'green': 'prob',
    'yellow': 'prob',
    'red': 'prob',
    # Evolution
    'green-yellow': 'int_tuple',
    'yellow-red': 'int_tuple',
    'red-fire': 'int_tuple',
    'fire-green': 'int_tuple'
}

HEATMAP_COLORS = {
    'green': (0, 250, 0),
    'yellow': (255, 204, 0),
    'red': (250, 0, 0),
    'fire': (0, 0, 0),
    # Biome locked cell color
    'mountain': (153, 102, 51) # brown
}

HEATMAP_TRANSITION_GUIDE = ('green', 'yellow', 'red', 'fire')

# Environment engine configuration
FPS = 15 # frames per second (in Hz)

#---------------------------------
# Sprites
#---------------------------------
# Filepaths to the sprites themselves
MOUNTAIN_SPRITE_FILEPATH = os.path.join(PATH, '../../mountain.png')
FIRE_SPRITE_FILEPATH = os.path.join(PATH, '../../fire.png')

CHARACTER_SPRITE_FILEPATH = os.path.join(PATH, '../../drone.png')

#---------------------------------
# class Environment
#---------------------------------
class Environment:
    def __init__(self, config_file):
        # Load configuration for an environment
        self.__config = self.__parse_config(config_file)

        self.__screen = None
        self.__env_mtrx_repr = []
        self.__characters = []

        # Build the environment
        # (heatmap gridworld with heat signatures and biome elements)
        self.__build()

    def __parse_config(self, filename):
        if not re.search('.*\.json', filename):
            raise ValueError('Config must be a .json file')

        try:
            with open(filename, 'r') as config_file:
                content = json.load(config_file)
        except:
            raise ValueError('Config file IO failure')

        # Error handle the json content and change datatypes inplace
        for field in CONFIG_FIELDS:
            if field not in content:
                raise ValueError(
                    'Missing config field \'{}\''.format(field)
                )

            # Error check tthe field
            is_valid, f_transform = self.__config_check(
                field, content[field])
            if not is_valid:
                raise ValueError(
                    'Invalid config value for \'{}\''.format(field)
                )
            else:
                content[field] = f_transform(content[field])

        # Isolate the window dimensions
        self.__WINDOW_WIDTH, self.__WINDOW_HEIGHT = content['grid_dim']

        # Check compatibility of window dimensions and cell_size
        cell_size = content['cell_size']

        if not (
            self.__WINDOW_WIDTH % cell_size == 0 and \
            self.__WINDOW_HEIGHT % cell_size == 0
        ):
            raise ValueError(
                '\'cell_size\' and \'grid_dim\' are incompatible'
            )

        # Check the fulfillement of the law of total probability
        total_prob = 0
        for color in ('green', 'yellow', 'red', 'mountain'):
            total_prob += content[color]

        if round(total_prob, 2) != 1:
            raise ValueError('Violated law of total probability')

        return content

    def __config_check(self, field, value):
        data_type = CONFIG_FIELDS[field]

        if data_type == 'prob':
            try:
                is_valid = 0 <= float(value) <= 1
            except:
                is_valid = False

            f_transform = lambda x : float(x)
        elif data_type == 'int':
            try:
                is_valid = int(value) > 1
            except:
                is_valid = False

            f_transform = lambda x : int(x)
        elif data_type == 'int_tuple':
            try:
                x, y = value.split(",")
                is_valid = int(x) > 1 and int(y) > 1
            except:
                is_valid = False

            f_transform = lambda x : list(map(int, x.split(',')))
        else:
            raise ValueError('Unknown data_type of \'{}\''.format(field))

        return is_valid, f_transform

    def __build(self):
        pygame.init()
        # Basic window configuration
        pygame.display.set_caption('AASMA Environment')
        self.__screen = pygame.display.set_mode(
            (self.__WINDOW_WIDTH, self.__WINDOW_HEIGHT)
        )

        # Draw the environment itself
        self.__draw_and_spawn_environment()

        # Draw the character
        self.__draw_and_spawn_character()

        # First screen render
        pygame.display.flip()

    def __draw_and_spawn_environment(self):
        # Load required sprites for later use
        cell_size = self.__config['cell_size']
        mountain = pygame.transform.scale(
            pygame.image.load(MOUNTAIN_SPRITE_FILEPATH),
            (cell_size - 2, cell_size - 2)
        )

        # This will eventually be (after heatmap placement)
        # the color of the cell sperating lines
        self.__screen.fill((0, 0, 0))

        # Init the heat map
        # Use the heatmap heat signature probabilities
        # provided in config.json

        # Heat signature probability distribution
        p_distribution = np.array([
            self.__config['green'],
            self.__config['yellow'],
            self.__config['red'],
            self.__config['mountain']
        ])
        choice = ('green', 'yellow', 'red', 'mountain')

        # Populate each cell with a heat signature and biome elements
        for y in range(0, self.__WINDOW_WIDTH, cell_size):
            row = []
            for x in range(0, self.__WINDOW_HEIGHT, cell_size):
                # Create the heat signature cell
                do_draw = True
                while do_draw:
                    # Assume it's a one time draw
                    do_draw = False
                    # Draw the color
                    color = np.random.choice(choice, 1, p=p_distribution)[0]

                    if color == 'mountain':
                        # Mountain mirroring avoidance
                        do_draw = not self.__check_good_mountain_pos(
                            x//cell_size, y//cell_size
                        )

                # Create a heatmap tile ((cell_size-4) * (cell_size-4))
                # and place it in the environment
                pygame.draw.rect(
                    self.__screen, HEATMAP_COLORS[color],
                    pygame.Rect(
                        x + 2, y + 2,
                        cell_size - 2, cell_size - 2
                    ),
                    0 # 0 = fill cell
                )

                # Add cell to environment matrix representation
                # cell = [color, time step]
                row.append([
                    color,
                    0 if color == 'mountain' else \
                    randint(*self.__config[
                        color + '-' + self.__get_next_evolution_color(color)
                    ])
                ])

                # Sprite injection
                # The only sprite to be inserted at this stage is the mountain
                # Fires don't exist in the beginning
                # Characters are not spawned here
                if color == 'mountain':
                    self.__screen.blit(mountain, (x + 2, y + 2))

            # Add entire row to the matrix representation
            self.__env_mtrx_repr.append(row)

    def __get_next_evolution_color(self, color):
        try:
            return HEATMAP_TRANSITION_GUIDE[
                (HEATMAP_TRANSITION_GUIDE.index(color) + 1) \
                % len(HEATMAP_TRANSITION_GUIDE)
            ]
        except:
            raise ValueError(
                'Can\'t update cell with color \'{}\''.format(color)
            )

    def __check_good_mountain_pos(self, x, y):
        if y == 0:
            return True

        # (1) and (4)
        try:
            pos_1 = self.__env_mtrx_repr[y-1][x+1][0] == 'mountain'
        except:
            return False

        # (1)
        try:
            pos_2 = self.__env_mtrx_repr[y-2][x][0] == 'mountain'
        except:
            pos_2 = False

        if pos_1 and pos_2:
            return False

        # (4)
        try:
            pos_2 = self.__env_mtrx_repr[y-1][x-1][0] == 'mountain'
        except:
            pos_2 = False

        if pos_1 and pos_2:
            return False

        # (2) and (3)
        try:
            pos_1 = self.__env_mtrx_repr[y-1][x-1][0] == 'mountain'
        except:
            return False

        # (2)
        try:
            pos_2 = self.__env_mtrx_repr[y-2][x][0] == 'mountain'
        except:
            pos_2 = False

        if pos_1 and pos_2:
            return False

        # (3)
        try:
            pos_2 = self.__env_mtrx_repr[y][x-2][0] == 'mountain'
        except:
            pos_2 = False

        if pos_1 and pos_2:
            return False

        return True

    def __draw_and_spawn_character(self):
        # Load character sprite
        cell_size = self.__config['cell_size']
        character = pygame.transform.scale(
            pygame.image.load(CHARACTER_SPRITE_FILEPATH),
            (cell_size - 2, cell_size - 2)
        )

        # Spawn a character at a random location
        # (without collision with other characters and biome elements)
        do_spawn = True
        while do_spawn:
            # Get a new random position (both in pixels and cell units)
            x_mtrx_i = randint(0, (self.__WINDOW_WIDTH // cell_size) - 1)
            x = x_mtrx_i * cell_size
            y_mtrx_i = randint(0, (self.__WINDOW_HEIGHT // cell_size) - 1)
            y = y_mtrx_i * cell_size

            # Collision avoidance
            # with biome elements...
            env_pos = self.__env_mtrx_repr[x_mtrx_i][y_mtrx_i]
            has_biome_object = env_pos[0] == 'mountain'

            # ... and with other characters
            has_character = False
            for agent in self.__characters:
                coll_x, coll_y = agent['x'], agent['y']
                has_character = (coll_x == x_mtrx_i) and (coll_y == y_mtrx_i)
                if has_character:
                    break

            if not (has_biome_object or has_character):
                do_spawn = False

        # Create a unique agent id
        now = datetime.now()
        cur_time = now.strftime("%m/%d/%Y%H:%M:%S.%f")
        agent_id = hashlib.sha1(cur_time.encode()).hexdigest()

        # The values of the previous are initialized with the spawn coordinates
        self.__characters.append({'id': agent_id, 'x': x, 'y': y, 'x_prev': x, 'y_prev': y, 'spawn_color': env_pos[0]})
        self.__screen.blit(character, (x + 2, y + 2))

    def __update_heatmap(self):
        cell_size = self.__config['cell_size']
        # Load the fire sprite once throughout this update
        fire = pygame.transform.scale(
            pygame.image.load(FIRE_SPRITE_FILEPATH),
            (cell_size - 2, cell_size - 2)
        )

        for x in range(0, self.__WINDOW_WIDTH, cell_size):
            x_mtrx_i = x // cell_size
            for y in range(0, self.__WINDOW_HEIGHT, cell_size):
                y_mtrx_i = y // cell_size

                color, ts = self.__env_mtrx_repr[x_mtrx_i][y_mtrx_i]

                if (ts <= 0) and (color != 'mountain'):
                    # Update heatmat tile
                    # Get the color of the current update
                    color = self.__get_next_evolution_color(color)

                    # Place the new heatmap tile in the screen
                    pygame.draw.rect(
                        self.__screen, HEATMAP_COLORS[color],
                        pygame.Rect(
                            x + 2, y + 2,
                            cell_size - 2, cell_size - 2
                        ),
                        0 # 0 = fill cell
                    )

                    if color == 'fire':
                        self.__screen.blit(fire, (x + 2, y + 2))

                    # Get the next color to be displayed
                    # (when time step resets)
                    next_color = self.__get_next_evolution_color(color)

                    # Update cell of matrix representation
                    self.__env_mtrx_repr[x_mtrx_i][y_mtrx_i] = [
                        color,
                        randint(*self.__config[color + '-' + next_color])
                    ]
                elif ts > 0:
                    # Passage passage of time
                    self.__env_mtrx_repr[x_mtrx_i][y_mtrx_i][1] -= 1

    def __update_character(self):
        # TODO
        # Reset when agent enters the cell
        for agent in self.__characters:
            #Get the pos in matrix form to return the current color
            cell_size = self.__config['cell_size']
            x_mtrx_i = agent['x'] // cell_size
            y_mtrx_i = agent['y'] // cell_size
            env_pos = self.__env_mtrx_repr[x_mtrx_i][y_mtrx_i]
            env_pos_color = env_pos[0]

            # Check if an agent has moved
            # Check if heatmap tile has been updated
            if (agent['x'] != agent['x_prev'] or agent['y'] != agent['y_prev'])\
                    or agent['spawn_color'] != env_pos_color:

                #Update the new spawn color
                agent['spawn_color'] = env_pos_color

                # Load character sprite
                cell_size = self.__config['cell_size']
                character = pygame.transform.scale(
                    pygame.image.load(CHARACTER_SPRITE_FILEPATH),
                    (cell_size - 2, cell_size - 2)
                )
                self.__screen.blit(character, (agent['x'] + 2, agent['y'] + 2))

    def run(self):
        # Start the clock
        clock = pygame.time.Clock()

        # Main loop
        while True:
            # Handle exit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        return

            # Update heatmap...
            self.__update_heatmap()
            # ...and character movement
            self.__update_character()

            pygame.display.flip()
            clock.tick(FPS)
