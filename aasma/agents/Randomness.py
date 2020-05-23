#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
# File : Randomness.py
#
# @ start date          21 05 2020
# @ last update         23 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import aasma.agents.AgentAbstract as AgentAbstract
import numpy as np

#---------------------------------
# class AgentAbstract
#---------------------------------
class Randomness(AgentAbstract.AgentAbstract):
    def __init__(self):
        super(Randomness, self).__init__()

    def make_action(self, state):
        return np.random.choice(self.actions)
