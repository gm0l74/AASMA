#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
# File : run.py
#
# @ start date          17 05 2020
# @ last update         24 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import sys, copy
from datetime import datetime
import numpy as np

import environment, pygame
import utils

import aasma.agents.Reactive as Reactive
import aasma.agents.Randomness as Randomness
import aasma.agents.DeepQ as DeepQ

#---------------------------------
# Environment Engine
#---------------------------------
FPS = 60

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python run.py [drl|random|reactive] [single|multi] <path>")
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

        # A single neural net can be used to control multiple agents
        agent = DeepQ.DeepQ(sys.argv[3])

        # Init the states fed to the neural net later on
        snapshot = utils.perceive(grabber.snapshot())

        if n_agents == 1:
            # There's only a single state
            state = np.array([snapshot for _ in range(4)])
        else:
            # There's are as many states as there are agents
            state = []
            _, characters = env.gods_view()
            for character in characters:
                position = [character['x'], character['y']]
                alt_snap = utils.remove_character_from_image(snapshot, position)

                state.append(np.array([alt_snap for _ in range(4)]))

    elif agent_type == 'random':
        agent = Randomness.Randomness()
    else:
        agent = Reactive.Reactive()

    clock = pygame.time.Clock()
    RUN = True

    while RUN:
        # Handle pygame events
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
            snapshot = utils.perceive(grabber.snapshot())

            if n_agents == 1:
                snapshot = snapshot.reshape(1, *utils.IMG_SIZE)
                state = np.concatenate((state[1:], snapshot), axis=0)

                env.move_character(0, utils.ACTIONS[
                    agent.predict(
                        np.expand_dims(np.transpose(state, [1, 2, 0]), axis=0)
                    )[1]
                ])
            else:
                _, characters = env.gods_view()
                characters = copy.deepcopy(characters)
                for i, character in enumerate(characters):
                    position = [character['x'], character['y']]

                    alt_snap = utils.remove_character_from_image(snapshot, position)
                    alt_snap = alt_snap.reshape(1, *utils.IMG_SIZE)

                    state[i] = np.concatenate((state[i][1:], alt_snap), axis=0)

                    env.move_character(1 if i == 0 else 0, utils.ACTIONS[
                        agent.predict(
                            np.expand_dims(
                                np.transpose(state[i], [1, 2, 0]), axis=0
                            )
                        )[1]
                    ])

        env.update()
        clock.tick(FPS)
