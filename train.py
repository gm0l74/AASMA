#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
# File : train.py
#
# @ start date          16 05 2020
# @ last update         18 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import pygame

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import environment, agent
import utils

#---------------------------------
# Training Parameters
#---------------------------------
N_EPISODES = 50
MAX_EPISODE_LENGTH = 200

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    # Init the environment and spawn a character
    env = environment.Environment('./config.json')
    env.add_character()

    import grabber

    # Spawn the agent
    dql_agent = agent.DeepQAgent()

    clock = pygame.time.Clock()
    RUN = True ; episode_i = 0
    while RUN and episode_i < N_EPISODES:
        print("EPISODE {}/{}".format(episode_i + 1, N_EPISODES))

        # Reset the environemnt and the initial state
        env.reset()
        current_state = utils.perceive(grabber.snapshot())

        in_episode_i = 0
        # Episode loop
        while RUN and in_episode_i < MAX_EPISODE_LENGTH:
            # Handle exit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    RUN = False

            action = dql_agent.make_action(current_state)

            # Propagate the action to the environment
            env.move_character(0, utils.ACTIONS[action])
            # ... and get the reward of said action
            reward = env.update()[0]
            print('[{}]\t{}\t[{}]'.format(
                datetime.now().strftime('%H:%M:%S'),
                utils.ACTIONS[action], reward
            ))

            # Retrieve the new state
            new_state = utils.perceive(grabber.snapshot())

            # Add this to the replay memory
            dql_agent.add_memory(current_state, action, reward, new_state)

            # Train models
            dql_agent.replay()
            dql_agent.target_train()

            current_state = new_state

            # Save model every 5 episodes
            if episode_i % 5 == 0:
                dql_agent.save()

            in_episode_i += 1
            clock.tick(15)

        episode_i += 1
