#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
# File : environment.py
#
# @ start date          16 05 2020
# @ last update         18 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import os, pygame, copy
import numpy as np
from random import randint

import utils

#---------------------------------
# Constants
#---------------------------------
PATH = os.path.realpath(__file__)
# Remove filename
PATH = PATH[:len(PATH) - 14]

# Changes behavior of environment reset
LOCK_RESET = True

# Environemnt operation
HEATMAP_COLORS = {
    'green': (0, 250, 0),
    'yellow': (255, 204, 0),
    'red': (250, 0, 0),
    'fire': (0, 0, 0),
    'mountain': (153, 102, 51) # brown
}

HEATMAP_TRANSITION_GUIDE = ('green', 'yellow', 'red', 'fire', 'yellow')

#---------------------------------
# Sprites
#---------------------------------
MOUNTAIN_SPRITE_FILEPATH = os.path.join(PATH, 'sprites/mountain.png')
FIRE_SPRITE_FILEPATH = os.path.join(PATH, 'sprites/fire.png')

CHARACTER_SPRITE_FILEPATH = os.path.join(PATH, 'sprites/drone.png')

#---------------------------------
# class Environment
#---------------------------------
class Environment:
    def __init__(self, config_file):
        # Load environment configuration
        self.__config = utils.parse_config(config_file)
        self.__WINDOW_WIDTH, self.__WINDOW_HEIGHT = self.__config['grid_dim']

        # Environment operation
        self.__screen = None
        self.__env_mtrx = []
        self.__reset_env_matrx = None

        # Character operation
        self.__characters = []
        self.__characters_reset = []

        self.__build()

    # --- Private functions ---
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

    def __init_matrix_repr(self):
        self.__env_mtrx = []

        # Heat signature probability distribution
        p_distribution = np.array([
            self.__config['green'],
            self.__config['yellow'],
            self.__config['red'],
            self.__config['mountain']
        ])
        c_choice = ('green', 'yellow', 'red', 'mountain')

        cell_size = self.__config['cell_size']
        for y in range(0, self.__WINDOW_WIDTH // cell_size):
            row_repr = []
            for x in range(0, self.__WINDOW_HEIGHT // cell_size):
                do_draw = True
                while do_draw:
                    # Assume it's a one time draw
                    do_draw = False
                    # Draw the color
                    color = np.random.choice(
                        c_choice, 1, p=p_distribution
                    )[0]

                    if color == 'mountain':
                        # Mountain mirroring avoidance
                        do_draw = not self.__check_good_mountain_pos(
                            x // cell_size, y // cell_size
                        )

                # Add cell to environment matrix representation
                row_repr.append([
                    color,
                    0 if color == 'mountain' else \
                    randint(*self.__config[
                        color + '-' + self.__get_next_evolution_color(color)
                    ])
                ])

            self.__env_mtrx.append(row_repr)

    def __draw_and_spawn_environment(self):
        cell_size = self.__config['cell_size']

        # Load required sprites
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
        self.__init_matrix_repr()
        self.__reset_env_matrx = copy.deepcopy(self.__env_mtrx)

        for y in range(0, self.__WINDOW_WIDTH, cell_size):
            for x in range(0, self.__WINDOW_HEIGHT, cell_size):
                color = self.__env_mtrx[y // cell_size][x // cell_size][0]
                self.__draw_heatmap_tile(y, x, color)

                # Sprite injection
                # The only sprite to be inserted at this stage is the mountain
                if color == 'mountain':
                    self.__screen.blit(mountain_sprite, (x + 2, y + 2))

    def __draw_heatmap_tile(self, y, x, color):
        cell_size = self.__config['cell_size']

        # Create a heatmap tile (cell_size-4) * (cell_size-4)
        # and place it in the environment
        pygame.draw.rect(
            self.__screen, HEATMAP_COLORS[color],
            pygame.Rect(
                x + 2, y + 2,
                cell_size - 2, cell_size - 2
            ),
            0 # 0 = fill cell
        )

        if color == 'fire':
            fire_sprite = pygame.transform.scale(
                pygame.image.load(FIRE_SPRITE_FILEPATH),
                (cell_size - 2, cell_size - 2)
            )
            self.__screen.blit(
                fire_sprite, (x + 2, y + 2)
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

        # Placements in first or last columns are not allowed
        if (
            x == 0 or \
            x == self.__WINDOW_WIDTH // self.__config['cell_size'] - 1
        ):
            return False

        # Layouts which are avoided by this algorithm
        # (all off the situations are evaluated at the
        # positions with greater value of y and x)
        #
        # Layouts:
        # (1) --M--  (2) --M--  (3) --M--  (4) ----
        #     -M---      ---M-      -M-M-      -M-M-
        #     --M--      --M--      -----      --M--

        # (2) and (4)
        try:
            pos_1 = self.__env_mtrx[y-1][x+1][0] == 'mountain'
        except:
            return False

        # (2)
        try:
            pos_2 = self.__env_mtrx[y-2][x][0] == 'mountain'
        except:
            pos_2 = False

        if pos_1 and pos_2:
            return False

        # (4)
        try:
            pos_2 = self.__env_mtrx[y-1][x-1][0] == 'mountain'
        except:
            pos_2 = False

        if pos_1 and pos_2:
            return False

        # (1) and (3)
        try:
            pos_1 = self.__env_mtrx[y-1][x-1][0] == 'mountain'
        except:
            return False

        # (1)
        try:
            pos_2 = self.__env_mtrx[y-2][x][0] == 'mountain'
        except:
            pos_2 = False

        if pos_1 and pos_2:
            return False

        # (3)
        try:
            pos_2 = self.__env_mtrx[y][x-2][0] == 'mountain'
        except:
            pos_2 = False

        if pos_1 and pos_2:
            return False

        return True

    def __update_heatmap(self):
        cell_size = self.__config['cell_size']

        # Load the fire sprite once throughout this update
        fire_sprite = pygame.transform.scale(
            pygame.image.load(FIRE_SPRITE_FILEPATH),
            (cell_size - 2, cell_size - 2)
        )

        for x in range(0, self.__WINDOW_WIDTH, cell_size):
            x_mtrx = x // cell_size
            for y in range(0, self.__WINDOW_HEIGHT, cell_size):
                y_mtrx = y // cell_size
                color, ts = self.__env_mtrx[y_mtrx][x_mtrx]

                if ts <= 0 and color != 'mountain':
                    # Update heatmat tile
                    color = self.__get_next_evolution_color(color)
                    self.__draw_heatmap_tile(y, x, color)

                    if color == 'fire':
                        self.__screen.blit(fire_sprite, (x + 2, y + 2))

                    # Update cell of matrix representation
                    next_color = self.__get_next_evolution_color(color)
                    self.__env_mtrx[y_mtrx][x_mtrx] = [
                        color,
                        randint(*self.__config[color + '-' + next_color])
                    ]
                elif ts > 0:
                    # Passage of time
                    self.__env_mtrx[y_mtrx][x_mtrx][1] -= 1

    def __update_characters(self):
        cell_size = self.__config['cell_size']
        reset_time = self.__config['search-time']

        # Load character sprite
        character_sprite = pygame.transform.scale(
            pygame.image.load(CHARACTER_SPRITE_FILEPATH),
            (cell_size - 2, cell_size - 2)
        )

        for character in self.__characters:
            x = character['x'] ; y = character['y']
            x_prev = character['x_prev'] ; y_prev = character['y_prev']

            curr_search_time = character['curr_search_time']
            hm_color = self.__env_mtrx[y][x][0]

            # Check if a character has moved
            if x != x_prev or y != y_prev:
                if curr_search_time >= reset_time:
                    # Character has searched the area. Now it's green!
                    try:
                        character['reward'] += self.__config[hm_color + '_dtct']
                    except:
                        character['reward'] += self.__config['invalid_pos']

                    self.__draw_heatmap_tile(
                        y * cell_size, x * cell_size, 'green'
                    )

                    # Override '__update_heatmap'
                    self.__env_mtrx[y][x] = [
                        'green',
                        randint(*self.__config[
                            'green-' + self.__get_next_evolution_color('green')
                        ])
                    ]

                self.__draw_heatmap_tile(
                    y_prev * cell_size, x_prev * cell_size,
                    self.__env_mtrx[y_prev][x_prev][0]
                )

                # Draw character on the new cell
                self.__screen.blit(
                    character_sprite,
                    (x * cell_size + 2, y * cell_size + 2)
                )
                character['hm_color'] = self.__env_mtrx[y][x][0]
                character['curr_search_time'] = 0
            else:
                character['curr_search_time'] += 1

                # Check if heatmap cell color can be updated
                if curr_search_time >= reset_time:
                    try:
                        hm_color = character['hm_color']
                        character['reward'] += self.__config[hm_color + '_dtct']
                    except:
                        character['reward'] += self.__config['invalid_pos']

                    # Character has searched the area. Now it's green!
                    self.__draw_heatmap_tile(
                        y * cell_size, x * cell_size, 'green'
                    )

                    # Override '__update_heatmap'
                    self.__env_mtrx[y][x] = [
                        'green',
                        randint(*self.__config[
                            'green-' + self.__get_next_evolution_color('green')
                        ])
                    ]

                    # Redraw character on that same cell
                    self.__screen.blit(
                        character_sprite,
                        (x * cell_size + 2, y * cell_size + 2)
                    )
                    character['hm_color'] = 'green'
                elif character['hm_color'] != hm_color:
                    # Redraw character on that same cell
                    self.__screen.blit(
                        character_sprite,
                        (x * cell_size + 2, y * cell_size + 2)
                    )
                    character['hm_color'] = hm_color

        return [character['reward'] for character in self.__characters]

    def __find_insertion_zone(self):
        cell_size = self.__config['cell_size']

        do_spawn = True
        while do_spawn:
            # Get a new random position (both in pixels and grid units)
            x_mtrx = randint(0, (self.__WINDOW_WIDTH // cell_size) - 1)
            x = x_mtrx * cell_size
            y_mtrx = randint(0, (self.__WINDOW_HEIGHT // cell_size) - 1)
            y = y_mtrx * cell_size

            # Collision avoidance
            # with biome elements...
            hm_color = self.__env_mtrx[y_mtrx][x_mtrx][0]
            has_biome_object = hm_color == 'mountain'

            # ... and other characters
            has_character = False
            for character in self.__characters:
                coll_x, coll_y = character['x'], character['y']
                has_character = (coll_x == x_mtrx) and (coll_y == y_mtrx)
                if has_character:
                    break

            if not (has_biome_object or has_character):
                do_spawn = False

        return x_mtrx, y_mtrx, hm_color

    # --- Public functions ---
    def reset(self):
        cell_size = self.__config['cell_size']

        # Reset matrix representation...
        if LOCK_RESET:
            # ...to the same inital state
            self.__env_mtrx = copy.deepcopy(self.__reset_env_matrx)
        else:
            # ...to a new inital state
            self.__init_matrix_repr()

        # Load required sprites
        mountain_sprite = pygame.transform.scale(
            pygame.image.load(MOUNTAIN_SPRITE_FILEPATH),
            (cell_size - 2, cell_size - 2)
        )
        character_sprite = pygame.transform.scale(
            pygame.image.load(CHARACTER_SPRITE_FILEPATH),
            (cell_size - 2, cell_size - 2)
        )

        for y in range(0, self.__WINDOW_WIDTH // cell_size):
            for x in range(0, self.__WINDOW_HEIGHT // cell_size):
                color = self.__env_mtrx[y][x][0]
                self.__draw_heatmap_tile(y * cell_size, x * cell_size, color)

                # Sprite injection
                if color == 'mountain':
                    self.__screen.blit(
                        mountain_sprite,
                        (x * cell_size + 2, y * cell_size + 2)
                    )

        # Reset characters
        self.__characters = []
        for character_id in range(len(self.__characters_reset)):
            if LOCK_RESET:
                self.__characters.append(
                    copy.deepcopy(self.__characters_reset[character_id])
                )
            else:
                x, y, hm_color = self.__find_insertion_zone()

                # Character data structure initialization
                # and first character render
                character = {
                    'reward': 0,
                    'curr_search_time': 0,
                    'x': x,
                    'y': y,
                    'x_prev': x,
                    'y_prev': y,
                    'hm_color': hm_color
                }

                self.__characters.append(character)

            x = self.__characters[character_id]['x']
            y = self.__characters[character_id]['y']

            self.__screen.blit(
                character_sprite,
                (x * cell_size + 2, y * cell_size + 2)
            )

        # Re-render first render
        pygame.display.flip()

    def add_character(self):
        cell_size = self.__config['cell_size']

        # Load character sprite
        character_sprite = pygame.transform.scale(
            pygame.image.load(CHARACTER_SPRITE_FILEPATH),
            (cell_size - 2, cell_size - 2)
        )

        # Spawn a character at a random location
        # (without collision with other characters or biome elements)
        x_mtrx, y_mtrx, hm_color = self.__find_insertion_zone()

        # Character data structure initialization
        # and first character render
        character = {
            'reward': 0,
            'curr_search_time': 0,
            'x': x_mtrx,
            'y': y_mtrx,
            'x_prev': x_mtrx,
            'y_prev': y_mtrx,
            'hm_color': hm_color
        }

        self.__characters.append(character)
        self.__characters_reset.append(copy.deepcopy(character))

        x = x_mtrx * cell_size
        y = y_mtrx * cell_size
        self.__screen.blit(character_sprite, (x + 2, y + 2))

        # Render environment
        pygame.display.flip()

        return len(self.__characters) - 1

    def move_character(self, character_id, action):
        # Movement guide for each action in every axis
        mov_guide = {
            'up': {
                'x': lambda x: x, 'y': lambda y: y - 1
            },
            'down': {
                'x': lambda x: x, 'y': lambda y: y + 1
            },
            'left': {
                'x': lambda x: x - 1, 'y': lambda y: y
            },
            'right': {
                'x': lambda x: x + 1, 'y': lambda y: y
            },
            'stay': {
                'x': lambda x: x, 'y': lambda y: y
            }
        }

        if action not in mov_guide:
            raise ValueError('Invalid Action')

        character = self.__characters[character_id]
        new_x_mtrx = mov_guide[action]['x'](character['x'])
        new_y_mtrx = mov_guide[action]['y'](character['y'])

        # Check if new position is possible
        #  - can't move outside of the world boundaries
        #  - can't move into fires or mountains
        # - can't collide with other characters
        is_position_valid = True
        if not (0 <= new_x_mtrx < len(self.__env_mtrx[0])):
            is_position_valid = False

        if not (0 <= new_y_mtrx < len(self.__env_mtrx)):
            is_position_valid = False

        # Make sure we don't keep going with erroneous coordinates
        if is_position_valid:
            hm_type = self.__env_mtrx[new_y_mtrx][new_x_mtrx][0]
            if hm_type in ('mountain', 'fire'):
                is_position_valid = False

            # Avoid character collision
            for other_id, other_character in enumerate(self.__characters):
                if other_id != character_id:
                    if (
                        other_character['x'] == new_x_mtrx and \
                        other_character['y'] == new_y_mtrx
                    ):
                        is_position_valid = False
                        break

        # If the new position is not possible...
        # the character returns to its previous position
        character['x_prev'] = character['x']
        character['y_prev'] = character['y']

        if not is_position_valid:
            character['reward'] = self.__config['invalid_pos']
        else:
            character['x'] = new_x_mtrx
            character['y'] = new_y_mtrx

    def update(self):
        # Update heatmap...
        self.__update_heatmap()
        # ...and character movement
        rewards = self.__update_characters()

        # Count colors to calculate general penalty
        color_c = {'green': 0, 'yellow': 0, 'red': 0, 'fire': 0}

        cell_size = self.__config['cell_size']
        for x in range(0, self.__WINDOW_WIDTH // cell_size):
            for y in range(0, self.__WINDOW_HEIGHT // cell_size):
                color = self.__env_mtrx[y][x][0]
                if color != 'mountain':
                    color_c[color] += 1

        # Reset rewards for next update
        for character_id in range(len(self.__characters)):
            self.__characters[character_id]['reward'] = 0

        penalty = color_c['green'] * self.__config['green_exst'] + \
            color_c['yellow'] * self.__config['yellow_exst'] + \
            color_c['red'] * self.__config['red_exst'] + \
            color_c['fire'] * self.__config['fire_exst']

        pygame.display.flip()
        return rewards, penalty
