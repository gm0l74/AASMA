#!/usr/bin/env python3
#---------------------------------
# AASMA
# File : h2h.py
#
# @ start date          17 05 2020
# @ last update         24 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import environment, pygame
import utils

import aasma.agents.Reactive as Reactive
import aasma.agents.Randomness as Randomness
import aasma.agents.DeepQ as DeepQ

#---------------------------------
# HeadToHead Config
#---------------------------------
ENV_RESET = 3
N_SCENARIOS = 30
MAX_N_STEPS = 500

#---------------------------------
# Environment Engine
#---------------------------------
FPS = 240

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('\'weights_file_path\' is missing')

    env = environment.Environment('./config.json', reset_env=ENV_RESET)
    env.add_character()
    env.reset() ; env.reset()

    import grabber

    AGENTS = ('drl', 'random', 'reactive', 'm-drl')

    clock = pygame.time.Clock()

    # Metrics
    scores = [] ; steps = []

    # Main comparison loop
    for scenario_i in range(N_SCENARIOS):
        print('SCENARIO {}/{}'.format(scenario_i + 1, N_SCENARIOS))

        scenario_scores = []
        scenario_steps = []

        # Run the same scenario for every agent
        for agent_id in AGENTS:
            # Prepare the environment for the new agent
            env.reset()

            # Spawn the brain
            if agent_id == 'drl':
                agent = DeepQ.DeepQ(sys.argv[1])

                snapshot = utils.perceive(grabber.snapshot())
                state = np.array([snapshot for _ in range(4)])
            elif agent_id == 'm-drl':
                agent = DeepQ.DeepQ(sys.argv[1])

                state = []
                _, characters = env.gods_view()
                for character in characters:
                    position = [character['x'], character['y']]
                    alt_snap = utils.remove_character_from_image(
                        snapshot, position
                    )

                    state.append(np.array([alt_snap for _ in range(4)]))

            elif agent_id == 'random':
                agent = Randomness.Randomness()
            else:
                agent = Reactive.Reactive()

            # Reset score
            score = 0 ; step = 0

            RUN = True
            while RUN and step <= MAX_N_STEPS:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Quit request ignored")

                if agent_id == 'drl':
                    snapshot = utils.perceive(grabber.snapshot())
                    snapshot = snapshot.reshape(1, *utils.IMG_SIZE)

                    state = np.concatenate((state[1:], snapshot), axis=0)
                    env.move_character(0, utils.ACTIONS[
                        agent.predict(
                            np.expand_dims(
                                np.transpose(state, [1, 2, 0]),
                                axis=0
                            )
                        )[1]
                    ])
                elif agent_id == 'random':
                    env.move_character(0, agent.make_action(None))
                elif agent_id == 'drl':
                    overview = env.gods_view()
                    env.move_character(0, agent.make_action(overview))
                else:
                    # M-DRL
                    snapshot = utils.perceive(grabber.snapshot())

                    _, characters = env.gods_view()
                    characters = copy.deepcopy(characters)

                    for i, character in enumerate(characters):
                        position = [character['x'], character['y']]

                        alt_snap = utils.remove_character_from_image(
                            snapshot, position
                        )
                        alt_snap = alt_snap.reshape(1, *utils.IMG_SIZE)

                        state[i] = np.concatenate(
                            (state[i][1:], alt_snap),
                            axis=0
                        )

                        env.move_character(1 if i == 0 else 0, utils.ACTIONS[
                            agent.predict(
                                np.expand_dims(
                                    np.transpose(state[i], [1, 2, 0]), axis=0
                                )
                            )[1]
                        ])

                # Obtain feedback from the environment
                if agent_id == 'm-drl':
                    [r1, r2], env_penalty = env.update()
                    r = (r1 + r2) / 2
                else:
                    [r], env_penalty = env.update()

                if env_penalty <= -1:
                    RUN = False
                score += r
                step += 1
                clock.tick(FPS)

            scenario_scores.append(score)
            scenario_steps.append(step)

        scores.append(scenario_scores)
        steps.append(scenario_steps)

        print("[{}] SCORES|  DRL {:.2f} RANDOM {:.2f} REACTIVE {:.2f}".format(
            datetime.now().strftime('%H:%M:%S'),
            scenario_scores[0], scenario_scores[1], scenario_scores[2]
        ))

    # Final results
    scores = np.array(scores)
    steps = np.array(steps)

    drl_scores = scores[:,0] ; drl_steps = steps[:,0]
    random_scores = scores[:,1] ; random_steps = steps[:,1]
    reactive_scores = scores[:,2] ; reactive_steps = steps[:,2]

    print("AVERAGE SCORES|  DRL {:.2f} RANDOM {:.2f} REACTIVE {:.2f}".format(
        np.average(drl_scores),
        np.average(random_scores),
        np.average(reactive_scores)
    ))

    # Plot scores
    eps = range(drl_scores.shape[0])
    plt.plot(eps, drl_scores, label='drl', color='C0')
    plt.plot(eps, random_scores, label='random', color='C1')
    plt.plot(eps, reactive_scores, label='reactive', color='C2')

    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.legend()
    plt.show()

    # Plot survivability
    plt.plot(eps, drl_steps, label='drl', color='C0')
    plt.plot(eps, random_steps, label='random', color='C1')
    plt.plot(eps, reactive_steps, label='reactive', color='C2')

    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.legend()
    plt.show()
