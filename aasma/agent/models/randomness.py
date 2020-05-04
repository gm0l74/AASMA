#!/usr/bin/env python3
#---------------------------------
# AASMA - Agent
# File : randomness.py
#
# @ start date          02 05 2020
# @ last update         04 05 2020
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
        super(DeepReinforcementLearning, self).__init__()

    def perceive(self, snapshot):
        pass

    def make_action(self):
        return np.random.choice(self.actions)
