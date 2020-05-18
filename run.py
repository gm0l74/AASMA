#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
# File : run.py
#
# @ start date          17 05 2020
# @ last update         18 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import sys, pygame
from datetime import datetime
import numpy as np

import environment
import agent as Agent
import utils

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('Insufficient number of parameters')

    agent_type = sys.argv[1]

    # Init the environment and spawn a character
    env = environment.Environment('./config.json')
    env.add_character()

    if agent_type == 'drl':
        if len(sys.argv) != 3:
            raise ValueError('\'weights_file\' is missing')

        import grabber
        agent = Agent.DeepQAgent(sys.argv[2])

    clock = pygame.time.Clock()
    RUN = True
    while RUN:
        # Handle exit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                RUN = False

        if agent_type == 'random':
            # Make a completely random action
            action = np.random.choice(utils.ACTIONS)
        elif agent_type == 'drl':
            snapshot = utils.perceive(grabber.snapshot())

            # Use exploitation (e-greedy)
            action = utils.ACTIONS[agent.make_action(state, force=True)]
        else:
            raise ValueError('Invalid agent type')

        print("[{}] {}".format(
            datetime.now().strftime('%H:%M:%S'), action
        ))

        # Notify environment engine
        env.move_character(0, action)
        env.update()

        clock.tick(15)
