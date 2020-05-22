#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
# File : train.py
#
# @ start date          16 05 2020
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
# function Plotting
#---------------------------------
def plotting():
    colors = ['C2', ['C1', 'C0']]
    fig, axes = plt.subplots(2, 1)
    axes = axes.flat
    fig.suptitle('AASMA')

    # Common axes labels
    fig.text(0.5, 0.04, 'Episodes', ha='center')
    fig.text(0.04, 0.5, 'Rewards', va='center', rotation='vertical')

    # Plot data
    for axes_id in range(len(colors)):
        if axes_id == 0:
            axes[axes_id].plot(
                SCORES, label='Rewards',
                linestyle='solid', color=colors[axes_id]
            )
        else:
            axes[axes_id].plot(
                MIN_REWARDS, label='Min rewards',
                linestyle='solid', color=colors[axes_id][0]
            )

            axes[axes_id].plot(
                MAX_REWARDS, label='Max rewards',
                linestyle='solid', color=colors[axes_id][1]
            )

        # Styling
        axes[axes_id].legend()
        axes[axes_id].grid(True, linestyle='--')

    plt.show()

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    atexit.register(plotting)

    env = environment.Environment('./config.json')
    env.add_character()

    import grabber
    agent = DeepQ.DeepQ()

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

            # Exploitation
            q_values, action = agent.predict(
                np.expand_dims(np.transpose(state, [1, 2, 0]), axis=0)
            )

            # Exploration
            if np.random.random() < agent.epsilon:
                action =  np.random.choice(len(utils.ACTIONS))

            # Propagate the action to the environment
            env.move_character(0, utils.ACTIONS[action])

            # ... and get the reward of said action
            [r], env_penalty = env.update()
            r = env_penalty if env_penalty < 0 else r
            is_done = env_penalty < 0

            # Update metrics
            min_reward = r if min_reward is None else min(min_reward, r)
            max_reward = r if max_reward is None else max(max_reward, r)
            score += r + env_penalty

            # Get the result of making said action...
            snapshot = utils.perceive(grabber.snapshot())
            # ... and update the state
            next_state = np.concatenate((state[1:], snapshot), axis=None)

            # Insert new transition
            agent.feed_batch((
                q_values, state, action, next_state,
                np.sign(r), int(not is_done)
            ))

            # Update the current state for the next iteration
            state = next_state

            # Train the agent
            if (len(agent.batch) >= agent.mini_batch_size) and \
                (n_snapshots % 4 == 0):
                agent.train(is_done)
                agent.update_epsilon()

            clock.tick(FPS)

        episode_i += 1
        agent.episode += 1

        # Save the metrics of the finished episode
        SCORES.append(score)
        MIN_REWARDS.append(min_reward)
        MAX_REWARDS.append(max_reward)

        print('Episode: {}, Score: {}, Epsilon: {:.7f}, Alpha {:.7f}'.format(
            episode_i, score, agent.epsilon, agent.alpha
        ))

        # Update the target network every now and then
        if episode_i % UPDT_TARGET_NETWORK_FREQ == 0:
            agent.update_target_network()

        if episode_i % SAVE_FREQ == 0:
            agent.save_weights()
