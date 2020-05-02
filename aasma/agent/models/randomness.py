#!/usr/bin/env python3
#---------------------------------
# AASMA - Agent
# File : randomness.py
#
# @ start date          02 05 2020
# @ last update         02 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import numpy as np
from AgentModel import AgentModel

#---------------------------------
# class Randomness
#---------------------------------
class Randomness(AgentModel):
    def __init__(self, actions):
        if (not isinstance(actions, (list, tuple))) or len(actions) == 0:
            raise ValueError('Invalid set of actions')

        self.__actions = actions

    def perceive(self):
        pass

    def make_action(self):
        return np.random.choice(actions)
