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
import re, json
import pygame

#---------------------------------
# Constants
#---------------------------------
CONFIG_FIELDS = {
    'mountain': 'prob',
    'green': 'prob',
    'yellow': 'prob',
    'red': 'prob',
    'green-yellow': 'ts',
    'yellow-red': 'ts',
    'red-fire': 'ts'
}

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
        elif data_type == 'ts':
            try:
                return int(value) > 1
            except:
                return False
        else:
            raise ValueError(f'Unknown data_type of \'{field}\'')

    def __build(self):
        # TODO
        pass

    def __spawn(self):
        # TODO
        pass

    def host(self):
        # TODO
        pass
