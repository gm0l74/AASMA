#!/usr/bin/env python3
#---------------------------------
# AASMA
# File : AgentAbstract.py
#
# @ start date          21 05 2020
# @ last update         21 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import utils

#---------------------------------
# class AgentAbstract
#---------------------------------
class AgentAbstract:
    def __init__(self):
        self.n_actions = len(utils.ACTIONS)
        self.actions = utils.ACTIONS

    def make_action(self, state):
        raise ValueError('Override \'make_action\'')
