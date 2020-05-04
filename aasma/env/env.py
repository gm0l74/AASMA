#!/usr/bin/env python3
#---------------------------------
# AASMA - Environment
# File : env.py
#
# @ start date          22 04 2020
# @ last update         03 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import os, re, json, copy
import hashlib

import time
from datetime import datetime

import pygame
import numpy as np
from random import randint

import zmq, threading

#---------------------------------
# PWD
#---------------------------------
PATH = os.path.realpath(__file__)
# Remove filename
PATH = PATH[:len(PATH) - 6]

#---------------------------------
# Thread Syncing
#---------------------------------
SEMAPHORE = threading.BoundedSemaphore(1)
CHARACTER_ACCESS = threading.Lock()

#---------------------------------
# Constants
#---------------------------------
CONFIG_FIELDS = {
    # Dimensionality
    'grid_dim': 'int_tuple', # int_tuple is (int > 0, int > 0)
    'cell_size': 'int+',
    # Spawn
    'mountain': 'prob', # prob belongs to [0, 1]
    'green': 'prob',
    'yellow': 'prob',
    'red': 'prob',
    # Evolution
    'green-yellow': 'int_tuple',
    'yellow-red': 'int_tuple',
    'red-fire': 'int_tuple',
    'fire-yellow': 'int_tuple',
    # Character mechanics
    'search-cell-time': 'int0',
    # Score rewards
    'sc_invalid_pos': 'int',
    'sc_green_detect': 'int',
    'sc_yellow_detect': 'int',
    'sc_red_detect': 'int',
    'sc_green_exist': 'int',
    'sc_yellow_exist': 'int',
    'sc_red_exist': 'int',
    'sc_fire_exist': 'int'
}

HEATMAP_COLORS = {
    'green': (0, 250, 0),
    'yellow': (255, 204, 0),
    'red': (250, 0, 0),
    'fire': (0, 0, 0),
    'mountain': (153, 102, 51) # brown
}

HEATMAP_TRANSITION_GUIDE = ('green', 'yellow', 'red', 'fire', 'yellow')

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

        # Environment operation
        self.__screen = None
        self.__env_mtrx_repr = []
        self.__characters = {}

        # Score system
        self.__score = 0
        self.__metrics = {
            'n_green': 0,
            'n_yellow': 0,
            'n_red': 0,
            'n_fire': 0
        }

        # Build the environment
        # (heatmap gridworld with heat signatures and biome elements)
        self.__build()

    #------------------------------------------------------------------
    # Configuration
    #------------------------------------------------------------------
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
        elif data_type in ('int+', 'int', 'int0'):
            if data_type == 'int+':
                try:
                    is_valid = int(value) > 1
                except:
                    is_valid = False
            elif data_type == 'int':
                try:
                    int(value) ; is_valid = True
                except:
                    is_valid = False
            elif data_type == 'int0':
                try:
                    is_valid = int(value) >= 0
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

    #------------------------------------------------------------------
    # Environment build and character spawning
    #------------------------------------------------------------------
    def __build(self):
        pygame.init()
        # Basic window configuration
        pygame.display.set_caption('AASMA Environment')
        self.__screen = pygame.display.set_mode(
            (self.__WINDOW_WIDTH, self.__WINDOW_HEIGHT)
        )

        # Draw the environment itself
        self.__draw_and_spawn_environment()

        # First screen render
        pygame.display.flip()

    def __draw_and_spawn_environment(self):
        cell_size = self.__config['cell_size']
        # Load required sprites for later use
        mountain_sprite = pygame.transform.scale(
            pygame.image.load(MOUNTAIN_SPRITE_FILEPATH),
            (cell_size - 2, cell_size - 2)
        )

        # This will eventually be (after heatmap placement)
        # the color of the cell separating lines
        self.__screen.fill((0, 0, 0))

        # Init the heat map
        # Use the heatmap heat signature probabilities
        # provided in config.json (held in self.__config)

        # Heat signature probability distribution
        p_distribution = np.array([
            self.__config['green'],
            self.__config['yellow'],
            self.__config['red'],
            self.__config['mountain']
        ])
        c_choice = ('green', 'yellow', 'red', 'mountain')

        # Populate each cell with a heat signature and biome elements
        for y in range(0, self.__WINDOW_WIDTH, cell_size):
            row_repr = []
            for x in range(0, self.__WINDOW_HEIGHT, cell_size):
                # Create the heat signature cell
                do_draw = True
                while do_draw:
                    # Assume it's a one time draw
                    do_draw = False
                    # Draw the color
                    color = np.random.choice(c_choice, 1, p=p_distribution)[0]

                    if color == 'mountain':
                        # Mountain mirroring avoidance
                        do_draw = not self.__check_good_mountain_pos(
                            x//cell_size, y//cell_size
                        )

                # Create a heatmap tile ((cell_size-4) * (cell_size-4))
                self.__draw_heatmap_tile(y, x, color)

                # Add cell to environment matrix representation
                # cell = [color, time step]
                row_repr.append([
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
                    self.__screen.blit(mountain_sprite, (x + 2, y + 2))
                else:
                    # Update score
                    self.__score += self.__config['sc_' + color + '_exist']
                    self.__metrics['n_' + color] += 1

            # Add entire row to the matrix representation
            self.__env_mtrx_repr.append(row_repr)

    def __draw_heatmap_tile(self, y, x, color):
        cell_size = self.__config['cell_size']

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
        # Placements in first row are always allowed
        if y == 0:
            return True

        # Layouts which are avoided by this algorithm
        # (All off the situations are evaluated at the positions with
        # greater value of y and x)
        # Layouts:
        # (1) --M--  (2) --M--  (3) --M--  (4) ----
        #     -M---      ---M-      -M-M-      -M-M-
        #     --M--      --M--      -----      --M--

        # Solve those layouts
        # (2) and (4)
        try:
            pos_1 = self.__env_mtrx_repr[y-1][x+1][0] == 'mountain'
        except:
            return False

        # (2)
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

        # (1) and (3)
        try:
            pos_1 = self.__env_mtrx_repr[y-1][x-1][0] == 'mountain'
        except:
            return False

        # (1)
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
        cell_size = self.__config['cell_size']
        # Load character sprite
        character_sprite = pygame.transform.scale(
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
            env_pos = self.__env_mtrx_repr[y_mtrx_i][x_mtrx_i]
            has_biome_object = env_pos[0] == 'mountain'

            # ... and with other characters
            has_character = False
            for character in self.__characters.values():
                coll_x, coll_y = character['x'], character['y']
                has_character = (coll_x == x_mtrx_i) and (coll_y == y_mtrx_i)
                if has_character:
                    break

            if not (has_biome_object or has_character):
                do_spawn = False

        # Create a unique character id
        now = datetime.now()
        cur_time = now.strftime("%m/%d/%Y%H:%M:%S.%f")
        character_id = hashlib.sha1(cur_time.encode()).hexdigest()

        # Character data structure initialization
        if character_id in self.__characters:
            # Really unlikely
            print('Couldn\'t deploy character {}'.format(character_id))
            return 'nack'

        self.__characters[character_id] = {
            'id': character_id,
            'score': 0,
            'curr_search_time': 0,
            'x': x_mtrx_i, 'y': y_mtrx_i,
            'x_prev': x_mtrx_i, 'y_prev': y_mtrx_i,
            'hm_color': env_pos[0]
        }
        self.__screen.blit(character_sprite, (x + 2, y + 2))

        return character_id

    #------------------------------------------------------------------
    # Environment and character updates
    #------------------------------------------------------------------
    def __update_heatmap(self):
        cell_size = self.__config['cell_size']

        # Load the fire sprite once throughout this update
        fire_sprite = pygame.transform.scale(
            pygame.image.load(FIRE_SPRITE_FILEPATH),
            (cell_size - 2, cell_size - 2)
        )

        for x in range(0, self.__WINDOW_WIDTH, cell_size):
            x_mtrx_i = x // cell_size
            for y in range(0, self.__WINDOW_HEIGHT, cell_size):
                y_mtrx_i = y // cell_size

                color, ts = self.__env_mtrx_repr[y_mtrx_i][x_mtrx_i]

                if (ts <= 0) and (color != 'mountain'):
                    # Update heatmat tile
                    # Get the color of the current update
                    color = self.__get_next_evolution_color(color)

                    # Place the new heatmap tile in the screen
                    self.__draw_heatmap_tile(y, x, color)

                    # Update score
                    self.__score += self.__config['sc_' + color + '_exist']
                    self.__metrics['n_' + color] += 1

                    if color == 'fire':
                        self.__screen.blit(fire_sprite, (x + 2, y + 2))

                    # Get the next color to be displayed
                    # (when time step resets)
                    next_color = self.__get_next_evolution_color(color)

                    # Update cell of matrix representation
                    self.__env_mtrx_repr[y_mtrx_i][x_mtrx_i] = [
                        color,
                        randint(*self.__config[color + '-' + next_color])
                    ]
                elif ts > 0:
                    # Passage passage of time
                    self.__env_mtrx_repr[y_mtrx_i][x_mtrx_i][1] -= 1

    def __redraw_re_updated_heatmap_tile(self, y, x, color):
        # Used in '__update_character' which happens after '__update_heatmap'
        cell_size = self.__config['cell_size']

        # Load the fire sprite
        fire_sprite = pygame.transform.scale(
            pygame.image.load(FIRE_SPRITE_FILEPATH),
            (cell_size - 2, cell_size - 2)
        )

        # Place the heatmap tile in the screen without the drone sprite
        self.__draw_heatmap_tile(y  * cell_size, x  * cell_size, color)

        if color == 'fire':
            self.__screen.blit(
                fire_sprite, (x * cell_size + 2, y * cell_size + 2)
            )

    def __update_character(self):
        cell_size = self.__config['cell_size']
        time_to_reset_cell = self.__config['search-cell-time']

        # Load character sprite
        character_sprite = pygame.transform.scale(
            pygame.image.load(CHARACTER_SPRITE_FILEPATH),
            (cell_size - 2, cell_size - 2)
        )

        for character in self.__characters.values():
            # Current position
            x_mtrx_i = character['x']
            y_mtrx_i = character['y']

            # Previous position
            x_prev_mtrx_i = character['x_prev']
            y_prev_mtrx_i = character['y_prev']

            curr_search_time = character['curr_search_time']

            # Check if a character has moved
            if (x_mtrx_i != x_prev_mtrx_i) or (y_mtrx_i != y_prev_mtrx_i):
                hm_color = self.__env_mtrx_repr[y_prev_mtrx_i][x_prev_mtrx_i][0]

                # Check if heatmap cell color can be updated
                if curr_search_time > self.__config['search-cell-time']:
                    # Update character score TODO
                    character['score'] += \
                        self.__config['sc_' + hm_color + '_detect']

                    # Character has searched the area. Now it's green!
                    self.__redraw_re_updated_heatmap_tile(
                        y_prev_mtrx_i, x_prev_mtrx_i, 'green'
                    )
                else:
                    self.__redraw_re_updated_heatmap_tile(
                        y_prev_mtrx_i, x_prev_mtrx_i, hm_color
                    )

                # Draw character on the new cell
                self.__screen.blit(
                    character_sprite,
                    (x_mtrx_i * cell_size + 2, y_mtrx_i * cell_size + 2)
                )
                character['hm_color'] = hm_color
            else:
                hm_color = self.__env_mtrx_repr[y_mtrx_i][x_mtrx_i][0]

                # Check if heatmap cell color can be updated
                if curr_search_time > self.__config['search-cell-time']:
                    # Update character score TODO
                    character['score'] += \
                        self.__config['sc_' + hm_color + '_detect']

                    print("FUCK MY LIFE")
                    # Character has searched the area. Now it's green!
                    self.__redraw_re_updated_heatmap_tile(
                        y_mtrx_i, x_mtrx_i, 'green'
                    )

                    # Redraw character on that same cell
                    self.__screen.blit(
                        character_sprite,
                        (x_mtrx_i * cell_size + 2, y_mtrx_i * cell_size + 2)
                    )
                    character['hm_color'] = 'green'
                elif character['hm_color'] != hm_color:
                    # Redraw character on that same cell
                    self.__screen.blit(
                        character,
                        (x_mtrx_i * cell_size + 2, y_mtrx_i * cell_size + 2)
                    )
                    character['hm_color'] = hm_color

                character['curr_search_time'] += 1

    def __move_character(self, id, action):
        # Movement guide for each action and each axis
        mov_guide = {
            'up': {
                'x': lambda x: x,
                'y': lambda y: y - 1
            },
            'down': {
                'x': lambda x: x,
                'y': lambda y: y + 1
            },
            'left': {
                'x': lambda x: x - 1,
                'y': lambda y: y
            },
            'right': {
                'x': lambda x: x + 1,
                'y': lambda y: y
            },
            'stay': {
                'x': lambda x: x,
                'y': lambda y: y
            }
        }

        if action not in mov_guide:
            raise ValueError('Invalid Action')

        # Create new position
        character = self.__characters[id]
        new_x_mtrx = mov_guide[action]['x'](character['x'])
        new_y_mtrx = mov_guide[action]['y'](character['y'])

        # Check if new position is possible
        #  - can't move outside of the world boundaries
        #  - can't move into fires or mountains
        is_position_valid = True
        if not (0 <= new_x_mtrx < len(self.__env_mtrx_repr[0])):
            is_position_valid = False

        if not (0 <= new_y_mtrx < len(self.__env_mtrx_repr)):
            is_position_valid = False

        color = self.__env_mtrx_repr[new_y_mtrx][new_x_mtrx][0]
        if color in ('mountain', 'fire'):
            is_position_valid = False

        # If the new position is not possible...
        # the character returns to the original position
        # and a penalty is added to the score
        if not is_position_valid:
            self.__characters[id]['score'] += self.__config['sc_invalid_pos']
        else:
            # Update position
            self.__characters[id]['x_prev'] = self.__characters[id]['x']
            self.__characters[id]['y_prev'] = self.__characters[id]['y']
            self.__characters[id]['x'] = new_x_mtrx
            self.__characters[id]['y'] = new_y_mtrx

    #------------------------------------------------------------------
    # Environment execution and threading
    #------------------------------------------------------------------
    def __communicator(self):
        # Create the structure for inter-process communication
        socket = zmq.Context().socket(zmq.REP)
        socket.bind("tcp://*:5555")

        poller = zmq.Poller()
        poller.register(socket)
        n_character_threads = 0

        # Relay structure
        relay_characters = copy.deepcopy(self.__characters)

        while True:
            # Sync with main thread
            if SEMAPHORE.acquire(False):
                # Sync threads and update character intelligence structure
                self.__characters = copy.deepcopy(relay_characters)
            else:
                ready = dict(poller.poll(n_character_threads))
                if ready.get(socket):
                    # Receive transmission from character client
                    message = socket.recv().decode()
                    print(" <-- {}".format(message))

                    # Handle request
                    message = message.split(',')
                    if message[0] == 'create':
                        n_character_threads += 1
                        #CHARACTER_ACCESS.acquire()
                        response = self.__draw_and_spawn_character()
                        #CHARACTER_ACCESS.release()
                    elif message[0] == 'move':
                        #CHARACTER_ACCESS.acquire()
                        try:
                            self.__move_character(message[1], message[2])
                        except:
                            pass
                        #CHARACTER_ACCESS.release()
                        response = 'ok'
                    else:
                        raise ValueError('Couldn\'t handle message')

                    # Send response back to character client
                    socket.send(response.encode())
                    print(" --> {}".format(response))

    def run(self):
        # Start the engine clock and temporal variables
        clock = pygame.time.Clock() ; n_ticks = 0
        t_seconds = 0

        # Create another thread to handle inter-process communication (ipc)
        ipcThread = threading.Thread(target=self.__communicator)

        # Set ipc to daemon
        # Once the main thread is done, the daemon thread will be killed
        ipcThread.daemon = True
        ipcThread.start()

        # Main loop
        RUN_GAME = True
        while RUN_GAME:
            # Handle exit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    RUN_GAME = False

            # Update heatmap...
            self.__update_heatmap()
            # ...and character movement
            #CHARACTER_ACCESS.acquire()
            self.__update_character()
            #CHARACTER_ACCESS.release()

            # Show the score every second
            if n_ticks % FPS == 0:
                score = self.__score
                for character in self.__characters.values():
                    score += character['score']
                print("-- Tick {}s\tScore {}".format(t_seconds, score))
                t_seconds += 1

            pygame.display.flip()
            clock.tick(FPS) ; n_ticks += 1

            # Sync with the communicator
            try:
                SEMAPHORE.release()
            except ValueError:
                time.sleep(1/FPS)

        # Display metrics
        print("\n--- Metrics ---")
        for k, v in self.__metrics.items():
            print("{} - {}".format(k, v))
        print("---------------")
