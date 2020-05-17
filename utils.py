#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
# File : utils.py
#
# @ start date          16 05 2020
# @ last update         17 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import re, json
from PIL import Image
import numpy as np

#---------------------------------
# Constants
#---------------------------------
ACTIONS = ('up', 'down', 'left', 'right') # 'stay'
IMG_SIZE = (100, 100)

CONFIG_FIELDS = {
    # Dimensionality
    'grid_dim': 'int_tuple', # int_tuple is (int > 1, int > 1)
    'cell_size': 'int+',
    # Spawn
    'mountain': 'prob',
    'green': 'prob',
    'yellow': 'prob',
    'red': 'prob',
    # Evolution
    'green-yellow': 'int_tuple',
    'yellow-red': 'int_tuple',
    'red-fire': 'int_tuple',
    'fire-yellow': 'int_tuple',
    # Character mechanics
    'search-time': 'int0',
    # Reward system
    'invalid_pos': 'int',
    'green_dtct': 'int',
    'yellow_dtct': 'int',
    'red_dtct': 'int'
}

#---------------------------------
# Utility functions
#---------------------------------
def parse_config(filename):
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
        is_valid, f_transform = config_field_check(
            field, content[field])
        if not is_valid:
            raise ValueError(
                'Invalid config value for \'{}\''.format(field)
            )
        else:
            content[field] = f_transform(content[field])

    # Isolate the window dimensions
    WINDOW_WIDTH, WINDOW_HEIGHT = content['grid_dim']

    # Check compatibility of window dimensions and cell_size
    cell_size = content['cell_size']

    if not (
        WINDOW_WIDTH % cell_size == 0 and \
        WINDOW_HEIGHT % cell_size == 0
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

def config_field_check(field, value):
    data_type = CONFIG_FIELDS[field]
    # Supported data_types: prob, int+, int, int0, int_tuple
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

def perceive(snap):
    image = Image.fromarray(snap, 'RGB').convert('L').resize(IMG_SIZE)

    # Convert to a numpy array
    return np.asarray(
        image.getdata(), dtype=np.uint8
    ).reshape(image.size[1], image.size[0])
