#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
# File : run.py
#
# @ start date          17 05 2020
# @ last update         22 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import sys
from datetime import datetime
import numpy as np

import environment, pygame
import utils

import agents.Reactive as Reactive
import agents.Randomness as Randomness
import agents.DeepQ as DeepQ

#---------------------------------
# Environment Engine
#---------------------------------
FPS = 120

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usager: python run.py [drl|random|reactive] [single|multi]")
        raise ValueError('Insufficient number of parameters')

    agent_type = sys.argv[1]
    system_type = sys.argv[2]

    # Error handling
    if agent_type not in ('drl', 'random', 'reactive'):
        raise ValueError('Invalid agent type')

    if system_type not in ('single', 'multi'):
        raise ValueError('Invalid system type')

    # Init the environment
    env = environment.Environment('./config.json')

    # Spawn the desired number of characters
    n_agents = 1 if system_type == 'single' else 2
    for _ in range(n_agents):
        env.add_character()

    # Create the brains of the agents
    if agent_type == 'drl':
        if len(sys.argv) != 4:
            raise ValueError('\'weights_file_path\' is missing')

        import grabber

        agent = DeepQ(sys.argv[3])
        state = np.array([ utils.perceive(grabber.snapshot()) for _ in range(4)])
    elif agent_type == 'random':
        agent = Randomness()
    else:
        agent = Reactive()

    clock = pygame.time.Clock()
    RUN = True

    while RUN:
        # Handle pygame events (Exit)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                RUN = False

        if agent_type == 'random':
            for agent_id in range(n_agents):
                env.move_character(agent_id, agent.make_action(None))
        elif agent_type == 'reactive':
            for agent_id in range(n_agents):
                overview = env.gods_view()
                env.move_character(agent_id, agent.make_action(overview))
        else:
            # Deep Reinforcement Agent
            snapshot = utils.perceibe(grabber.snapshot())
            state = np.concatenate((state[1:], snapshot), axis=None)

            for agent_id in range(n_agents):
                env.move_character(agent_id, agent.make_action(state))

        print("[{}] SCORE {}".format(
            datetime.now().strftime('%H:%M:%S'),
        ))

        env.update()
        clock.tick(FPS)
