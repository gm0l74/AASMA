#!/usr/bin/env python3
#---------------------------------
# AASMA - Agent
# File : AgentModel.py
#
# @ start date          02 05 2020
# @ last update         02 05 2020
#---------------------------------

#---------------------------------
# class AgentModel
#---------------------------------
class AgentModel:
    def perceive(self):
        raise ValueError('Override function \'perceive\'')

    def make_action(self):
        raise ValueError('Override function \'make_action\'')
