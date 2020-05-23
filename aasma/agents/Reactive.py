#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
# File : Reactive.py
#
# @ start date          21 05 2020
# @ last update         23 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import numpy as np
import aasma.agents.AgentAbstract as AgentAbstract

#---------------------------------
# class Reactive
#---------------------------------
class Reactive(AgentAbstract.AgentAbstract):
    def __init__(self):
        super(Reactive, self).__init__()

    def make_action(self, gods_view):
        environment, agents = gods_view

        # Access agent's position
        x, y = agents[0]['x'], agents[0]['y']

        actions = {
            "up":       (x, y - 1),
            "down":     (x, y + 1),
            "left":     (x - 1, y),
            "right":    (x + 1, y)
        }

        # Remove all impossible actions
        delete = []
        for action in actions:
            if actions[action][0] < 0 or actions[action][0] > len(environment) - 1\
                or actions[action][1] < 0 or actions[action][1] > len(environment[0]) - 1:
                delete.append(action)
            elif environment[actions[action][1]][actions[action][0]][0] == "mountain":
                delete.append(action)
            elif environment[actions[action][1]][actions[action][0]][0] == "fire":
                delete.append(action)

        for key in delete: del actions[key]

        # See if they are all the same colour
        same_color = False
        if len(actions) > 0 :
            any_colour = list(actions.keys())[0]
            same_color = all(environment[actions[action][1]][actions[action][0]][0] ==
                environment[actions[any_colour][1]][actions[any_colour][0]][0] for action in actions)

            if same_color:
                return np.random.choice(list(actions.keys()))
            else:

                # See if it is ready
                list_reds = []
                for action in actions:
                    if environment[actions[action][1]][actions[action][0]][0] == "red":
                        list_reds.append(action)

                if len(list_reds) > 0:
                    return np.random.choice(list_reds)

                # See if anyone is yellow
                list_yellows = []
                for action in actions:
                    if environment[actions[action][1]][actions[action][0]][0] == "yellow":
                        list_yellows.append(action)

                if len(list_yellows) > 0:
                    return np.random.choice(list_yellows)

                raise ValueError('Reactive: Something went wrong')
        else:
            return np.random.choice(self.actions)
