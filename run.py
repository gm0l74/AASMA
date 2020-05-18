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
            print("ignitions:",env.get_ingnition_count())
        elif agent_type == 'drl':
            snapshot = utils.perceive(grabber.snapshot())
            #print(snapshot)
            # Use exploitation (e-greedy)
            q_values = agent.make_action(snapshot, force=True, all_values=True).copy()
            #print(q_values)
            sum_qvalues = np.sum(q_values)
            #print(sum_qvalues)
            for i in range(0, len(q_values)):
                q_values[i] = q_values[i]/sum_qvalues
            #print(q_values)
            action = utils.ACTIONS[np.random.choice(4, p=q_values)]
            #print(action)
            #action = utils.ACTIONS[
            #    np.argmax(agent.make_action(snapshot, force=True, all_values=True))
            #    ]
                
        else:
            raise ValueError('Invalid agent type')

        print("[{}] {}".format(
            datetime.now().strftime('%H:%M:%S'), action
        ))

        # Notify environment engine
        env.move_character(0, action)
        env.update()
        print("ignitions:",env.get_ingnition_count())

        clock.tick(15)
