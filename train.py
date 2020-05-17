#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
# File : train.py
#
# @ start date          16 05 2020
# @ last update         17 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import atexit
import pygame

import numpy as np
import matplotlib.pyplot as plt

import environment, agent
import utils

#---------------------------------
# Training Parameters
#---------------------------------
# Training parameters
N_EPISODES = 150
MAX_EPISODE_LENGTH = 20
UPDATE_FREQUENCY = 7
VALUENET_UPDATE_FREQ = 30
REPLAY_START_SIZE = 3

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    # Init the environment and spawn a character
    env = environment.Environment('./config.json')
    env.add_character()

    import grabber

    # Init the deep learning model
    agent = agent.DeepQAgent(utils.ACTIONS)

    # Weight file saving
    atexit.register(agent.save_progress)

    clock = pygame.time.Clock()
    RUN = True
    episode = 0
    while RUN and episode < N_EPISODES:
        print("EPISODE {}/{}".format(episode, N_EPISODES - 1))

        # Deep Learning Training
        env.reset()

        observation = utils.perceive(grabber.snapshot())
        # plt.imshow(observation)
        # plt.show()

        # Init state with the same observations
        state = np.array([ observation for _ in range(7) ])

        # Episode loop
        episode_step = 0

        while episode_step < MAX_EPISODE_LENGTH:
            # Handle exit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    RUN = False

            # Select an action using the agent
            action = agent.get_action(np.asarray([state]))

            # Make the action
            env.move_character(0, utils.ACTIONS[action])
            [reward] = env.update()

            observation = utils.perceive(grabber.snapshot())
            next_state = np.append(state[1:], [observation], axis=0)

            # Clip the reward
            clipped_reward = reward
            # Store transition in replay memory
            agent.add_experience(
                np.asarray([state]),
                action,
                clipped_reward,
                np.asarray([next_state])
            )

            # Train the agent
            do_update = episode_step % UPDATE_FREQUENCY == 0
            exp_check = len(agent.experiences) >= REPLAY_START_SIZE

            if do_update and exp_check:
                agent.train()

                # Every now and then, update ValueNet
                if agent.training_count % VALUENET_UPDATE_FREQ == 0:
                    print("Reset Value Net")
                    agent.reset_ValueNet()

            # Linear epsilon annealing
            if exp_check:
                agent.update_epsilon()

            # Prepare for the next iteration
            state = next_state

            episode_step += 1
            print(episode_step, utils.ACTIONS[action], reward)
            clock.tick(15)

        episode += 1
