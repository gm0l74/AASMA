#!/usr/bin/env python3
#---------------------------------
# AASMA - Agent
# File : AgentModel.py
#
# @ start date          02 05 2020
# @ last update         04 05 2020
#---------------------------------

#---------------------------------
# class AgentModel
#---------------------------------
class AgentModel:
    def __init__(self, actions):
        if (not isinstance(actions, (list, tuple))) or len(actions) == 0:
            raise ValueError('Invalid set of actions')
        
        self.actions = actions

    def perceive(self, snapshot):
        raise ValueError('Override function \'perceive\'')

    def make_action(self):
        raise ValueError('Override function \'make_action\'')
