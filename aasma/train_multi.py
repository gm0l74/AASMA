#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
# File : train_multi.py
#
# @ start date          22 05 2020
# @ last update         22 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import numpy as np
import utils
import environment, pygame
import atexit

import aasma.agents.DeepQ as DeepQ
import matplotlib.pyplot as plt

#---------------------------------
# Environment Engine
#---------------------------------
FPS = 120

#---------------------------------
# Training Settings
#---------------------------------
UPDT_TARGET_NETWORK_FREQ = 40
SAVE_FREQ = 100

#---------------------------------
# Metrics
#---------------------------------
SCORES = []
MIN_REWARDS = []
MAX_REWARDS = []

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('\'weights_file_path\' is missing')

    env = environment.Environment('./config.json')
    env.add_character()

    import grabber

    # Use transfer learning from single agent training
    # and restore training session
    agent1 = DeepQ.DeepQ(sys.argv[1]).restore()
    agent2 = DeepQ.DeepQ(sys.argv[1]).restore()

    episode_i = 0
    n_snapshots = 0

    clock = pygame.time.Clock()

    # Main training loop
    RUN = True ; episode_i = 0
    while RUN:
        env.reset()
        snapshot = utils.perceive(grabber.snapshot())
        state = np.array([snapshot for _ in range(4)])

        is_done = False

        # Metrics
        score = 0
        min_reward = None ; max_reward = None

        # Episode loop
        while (not is_done) and RUN:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    RUN = False

            n_snapshots += 1

            # Exploitation of agent1
            q_values, action = agent1.predict(
                np.expand_dims(np.transpose(state, [1, 2, 0]), axis=0)
            )

            # Exploration of agent1
            if np.random.random() < agent1.epsilon:
                action =  np.random.choice(len(utils.ACTIONS))

            # Propagate the action to the environment
            env.move_character(0, utils.ACTIONS[action])

            # Exploitation of agent2
            q_values, action = agent2.predict(
                np.expand_dims(np.transpose(state, [1, 2, 0]), axis=0)
            )

            # Exploration of agent2
            if np.random.random() < agent2.epsilon:
                action =  np.random.choice(len(utils.ACTIONS))

            # Propagate the action to the environment
            env.move_character(1, utils.ACTIONS[action])

            # ... and get the reward of said action
            [r1, r2], env_penalty = env.update()
            r1 = env_penalty if env_penalty < 0 else r1
            r2 = env_penalty if env_penalty < 0 else r2
            is_done = env_penalty < 0

            # Update metrics
            min_reward = (r1 + r2)/2 if min_reward is None \
                else min(min_reward, (r1 + r2)/2)
            max_reward = (r1 + r2)/2 if max_reward is None \
                else max(max_reward, (r1 + r2)/2)
            score += (r1 + r2 + env_penalty) / 2

            # Get the result of making said action...
            snapshot = utils.perceive(grabber.snapshot())
            # ... and update the state
            next_state = np.concatenate((state[1:], snapshot), axis=None)

            # Insert new transition
            agent1.feed_batch((
                q_values, state, action, next_state,
                np.sign(r1), int(not is_done)
            ))

            agent1.feed_batch((
                q_values, state, action, next_state,
                np.sign(r2), int(not is_done)
            ))

            # Update the current state for the next iteration
            state = next_state

            # Train the agent
            if (len(agent1.batch) >= agent1.mini_batch_size) and \
                (n_snapshots % 4 == 0):
                agent1.train(is_done)
                agent1.update_epsilon()

                agent2.train(is_done)
                agent2.update_epsilon()

            clock.tick(FPS)

        episode_i += 1
        agent1.episode += 1
        agent2.episode += 1

        # Save the metrics of the finished episode
        SCORES.append(score)
        MIN_REWARDS.append(min_reward)
        MAX_REWARDS.append(max_reward)

        print('Episode: {}, Score: {}, Epsilon: [{:.7f}, {:.7f}], Alpha {:.7f}'.format(
            episode_i, score, agent1.epsilon, agent2.epsilon, agent1.alpha
        ))

        # Update the target network every now and then
        if episode_i % UPDT_TARGET_NETWORK_FREQ == 0:
            agent1.update_target_network()
            agent2.update_target_network()

        if episode_i % SAVE_FREQ == 0:
            agent1.save_weights('agent1')
            agent2.save_weights('agent2')
