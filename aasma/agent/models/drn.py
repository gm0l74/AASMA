#!/usr/bin/env python3
#---------------------------------
# AASMA - Agent
# File : drn.py
#
# @ start date          22 04 2020
# @ last update         04 05 2020
#---------------------------------

# MEGA TODO

#---------------------------------
# Imports
#---------------------------------
import numpy as np
from AgentModel import AgentModel

#---------------------------------
# class DeepReinforcementLearning
#---------------------------------
class DeepReinforcementLearning(AgentModel):
    def __init__(self, actions):
        super(DeepReinforcementLearning, self).__init__()

    def perceive(self, snapshot):
        # TODO
        pass

    def make_action(self):
        # TODO
        return self.actions[0]
