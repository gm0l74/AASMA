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
from tqdm import tqdm

#---------------------------------
# Training Parameters
#---------------------------------
N_EPISODES = 150
MAX_EPISODE_LENGTH = 1000

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
        reward = 0
        while reward > -20:
            # Handle exit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    RUN = False

            action = dql_agent.make_action(current_state)

            # Propagate the action to the environment
            env.move_character(0, utils.ACTIONS[action])
            # ... and get the reward of said action
            rewards, penalty = env.update()
            reward = rewards[0] + penalty

            print('[{}]\t{}\t[{}]'.format(
                datetime.now().strftime('%H:%M:%S'),
                utils.ACTIONS[action], reward
            ))

            # Retrieve the new state
            new_state = utils.perceive(grabber.snapshot())

            # Add this to the replay memory
            dql_agent.add_memory(
                current_state, action, reward, new_state,
                reward <= -20
            )
            dql_agent.replay()

            current_state = new_state

            # Save model every 5 episodes
            if episode_i % 5 == 0:
                dql_agent.save()

            in_episode_i += 1

            if not RUN:
                break
            clock.tick(15)

        episode_i += 1
        dql_agent.target_train()

    dql_agent.save()
