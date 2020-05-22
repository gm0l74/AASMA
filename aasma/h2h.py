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
import matplotlib.pyplot as plt

import environment, pygame
import utils

import agents.Reactive as Reactive
import agents.Randomness as Randomness
import agents.DeepQ as DeepQ

#---------------------------------
# HeadToHead Config
#---------------------------------
ENV_RESET = 3
N_SCENARIOS = 20

#---------------------------------
# Environment Engine
#---------------------------------
FPS = 120

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('\'weights_file_path\' is missing')

    env = environment.Environment('./config.json', reset_env=ENV_RESET)
    import grabber

    AGENTS = ('drl', 'random', 'reactive')

    env.reset()

    clock = pygame.time.Clock()
    scores = []
    for scenario_i in range(N_SCENARIOS * ENV_RESET):
        scenario_scores = []
        scenario_steps = []

        # Run the same scenario for every agent
        for agent_id in AGENTS:
            # Prepare the environment for the new agent
            env.reset()

            # Spawn the brain
            if agent_id == 'drl':
                agent = DeepQ(sys.argv[1])

                snapshot = utils.perceive(grabber.snapshot())
                state = np.array([snapshot for _ in range(4)])
            elif agent_id == 'random':
                agent = Randomness()
            else:
                agent = Reactive()

            # Reset score
            score = 0 ; step = 0

            RUN = True
            while RUN:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Quit request ignored")

                if agent_id == 'drl':
                    snapshot = utils.perceibe(grabber.snapshot())
                    state = np.concatenate((state[1:], snapshot), axis=None)

                    env.move_character(agent_id, agent.make_action(state))
                elif agent_id == 'random':
                    env.move_character(agent_id, agent.make_action(None))
                else:
                    overview = env.gods_view()
                    env.move_character(agent_id, agent.make_action(overview))

                [r], _ = env.update()
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

    # Plot all scores
    plt.plot(drl_scores, label='drl', linestyle='solid', color='C0')
    plt.plot(random_scores, label='random', linestyle='solid', color='C1')
    plt.plot(reactive_scores, label='reactive', linestyle='solid', color='C2')

    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.legend()
    plt.show()

    # Plot survivability
    plt.plot(drl_steps, label='drl', linestyle='solid', color='C0')
    plt.plot(random_steps, label='random', linestyle='solid', color='C1')
    plt.plot(reactive_steps, label='reactive', linestyle='solid', color='C2')

    plt.xlabel('Episodes')
    plt.yabel('Steps')
    plt.show()
