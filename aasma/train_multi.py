#!/usr/bin/env python3
#---------------------------------
# AASMA
# File : train_multi.py
#
# @ start date          22 05 2020
# @ last update         24 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import sys, copy
import numpy as np
import utils
import environment, pygame
import atexit

import aasma.agents.DeepQ as DeepQ
import matplotlib.pyplot as plt

#---------------------------------
# Environment Engine
#---------------------------------
FPS = 360

#---------------------------------
# Training Settings
#---------------------------------
UPDT_TARGET_NETWORK_FREQ = 40
SAVE_FREQ = 100

RESTART_EPSILON = 0.8

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
    if len(sys.argv) != 2:
        raise ValueError('\'weights_file_path\' is missing')

    atexit.register(plotting)

    env = environment.Environment('./config.json')
    env.add_character()
    env.add_character()

    import grabber

    # Use transfer learning from single agent training
    # and restore training session
    agent = DeepQ.DeepQ(sys.argv[1])
    agent.restore_training(sys.argv[1])

    episode_i = agent.episode

    # Reset epsilon (exploration desire)
    agent.epsilon = RESTART_EPSILON

    n_snapshots = 0

    clock = pygame.time.Clock()

    # Main training loop
    RUN = True
    while RUN:
        env.reset()
        snapshot = utils.perceive(grabber.snapshot())

        # There's are as many states as there are agents
        state = []
        _, characters = env.gods_view()
        for character in characters:
            position = [character['x'], character['y']]
            alt_snap = utils.remove_character_from_image(snapshot, position)

            state.append(np.array([alt_snap for _ in range(4)]))

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

            _, characters = env.gods_view()
            characters = copy.deepcopy(characters)

            q_values_array = []
            actions_array = []
            for i, character in enumerate(characters):
                position = [character['x'], character['y']]

                alt_snap = utils.remove_character_from_image(
                    snapshot, position
                )
                alt_snap = alt_snap.reshape(1, *utils.IMG_SIZE)

                state[i] = np.concatenate((state[i][1:], alt_snap), axis=0)

                # Exploitation of agent
                q_values, action = agent.predict(
                    np.expand_dims(np.transpose(state[i], [1, 2, 0]), axis=0)
                )
                q_values_array.append(q_values)
                actions_array.append(action)

                # Exploration of agent
                if np.random.random() < agent.epsilon:
                    action =  np.random.choice(len(utils.ACTIONS))

                env.move_character(1 if i == 0 else 0, utils.ACTIONS[
                    agent.predict(
                        np.expand_dims(
                            np.transpose(state[i], [1, 2, 0]), axis=0
                        )
                    )[1]
                ])

            # ... and get the reward of said action
            [r1, r2], env_penalty = env.update()
            r1 = env_penalty if env_penalty < 0 else r1
            r2 = env_penalty if env_penalty < 0 else r2
            is_done = (env_penalty < 0) or (r1 + r2 <= -2)

            # Update metrics
            min_reward = (r1 + r2)/2 if min_reward is None \
                else min(min_reward, (r1 + r2)/2)
            max_reward = (r1 + r2)/2 if max_reward is None \
                else max(max_reward, (r1 + r2)/2)
            score += (r1 + r2 + env_penalty) / 2

            # Get the result of making said action...
            snapshot = utils.perceive(grabber.snapshot())
            # ... and update the state
            next_state = []
            for i, character in enumerate(characters):
                position = [character['x'], character['y']]

                alt_snap = utils.remove_character_from_image(
                    snapshot, position
                ).reshape(1, *utils.IMG_SIZE)

                next_state.append(np.concatenate(
                    (state[i][1:], alt_snap), axis=0
                ))

                # Insert new transition
                agent.add_experience((
                    q_values_array[i], state[i],
                    actions_array[i], next_state[i],
                    np.sign(r1 if i == 0 else r2), int(not is_done)
                ))

            # Update the current state for the next iteration
            state = next_state

            # Train the agent
            if (len(agent.memory) >= agent.mini_batch_size) and \
                (n_snapshots % 4 == 0):
                agent.learn()
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
            agent.save_weights('agent')
