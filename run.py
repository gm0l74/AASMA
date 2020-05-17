#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
# File : run.py
#
# @ start date          17 05 2020
# @ last update         17 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import sys
from datetime import datetime
import numpy as np
from PIL import Image

import environment, pygame
import agent as Agent

#---------------------------------
# Constants
#---------------------------------
ACTIONS = ('up', 'down', 'left', 'right', 'stay')
IMG_SIZE = (600, 600)

def perceive(snap):
    # Convert to gray-scale
    image = Image.fromarray(snap, 'RGB').convert('L').resize(IMG_SIZE)

    # Convert to a numpy array
    return np.asarray(
        image.getdata(), dtype=np.uint8
    ).reshape(image.size[1], image.size[0])

def state(last, observation):
    # Next state is composed by:
    # - last 3 snapshots of the previous state
    # - new observation
    return np.append(last[1:], [observation], axis=0)

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Wrong number of parameters')

    agent_type = sys.argv[1]

    # Init the environment and spawn a character
    env = environment.Environment('./config.json')
    env.add_character()

    if agent_type == 'drl':
        import grabber
        agent = Agent.DeepQAgent(
            ACTIONS, ['./policy_net.h5', './value_net.h5']
        )

        snapshot = perceive(grabber.snapshot())
        state = [snapshot for _ in range(4)]

    clock = pygame.time.Clock()
    RUN = True
    while RUN:
        # Handle exit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                RUN = False

        if agent_type == 'random':
            action = np.random.choice(ACTIONS)
        elif agent_type == 'drl':
            # Get the current state
            snapshot = perceive(grabber.snapshot())
            state = np.append(state[1:], [snapshot], axis=0)

            # Select an action using purely exploitation
            # print(agent.predict(state))
            action = ACTIONS[agent.predict(state)]
            print("[{}] {}".format(
                datetime.now().strftime('%H:%M:%S'), action
            ))
        else:
            raise ValueError('Invalid agent type')

        env.move_character(0, action)
        env.update()

        clock.tick(15)
