#!/usr/bin/env python3
#---------------------------------
# AASMA - Agent
# File : agent.py
#
# @ start date          22 04 2020
# @ last update         04 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import sys
import zmq, time
import matplotlib.pyplot as plt

import aasma.agent.grabber as grabber

import aasma.agent.models.drn as drn
import aasma.agent.models.randomness as randomness

#---------------------------------
# Constants
#---------------------------------
# Environment engine configuration
FPS = 15 # frames per second (in Hz)

DELAY = 1/FPS

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    # Error handle script arguments
    args = sys.argv
    if len(args) != 2:
        raise ValueError('Invalid number of arguments')

    # Error handle agent type
    agent_type = args[1]
    if agent_type not in ('randomness', 'drl'):
        raise ValueError('Invalid agent type')

    #  Connect to environment communicator
    print("Connecting to communicator...")
    ipc = zmq.Context().socket(zmq.REQ)

    ipc.setsockopt(zmq.LINGER, 0)
    ipc.setsockopt(zmq.AFFINITY, 1)
    ipc.setsockopt(zmq.RCVTIMEO, 3000) # 3 seconds timeout

    ipc.connect("tcp://localhost:5555")

    # Send spawn notification
    print("Sending spawn request")
    ipc.send(b"create")

    # Error handle the response
    try:
        agent_id = ipc.recv().decode()
    except:
        print("Unable to connect to environment")
        exit()

    if agent_id == 'nack':
        raise ValueError("Couldn't spawn agent")

    # Build the brains of the agent
    actions = ('up', 'down', 'left', 'right', 'stay')
    if agent_type == 'randomness':
        agent = randomness.Randomness(actions)
    else:
        agent = drl.DeepReinforcementLearning(actions)

    # Main cycle
    while True:
        try:
            # Perceive
            screen = grabber.snapshot()
            #plt.imshow(screen) ; plt.show() # debug
            agent.perceive(screen)

            # Reason on some action
            action = agent.make_action()

            # Send the selected action...
            print("Sending selected action ({})".format(action))
            query = "move,{},{}".format(agent_id, action)
            ipc.send(query.encode())

            #  ... and get the response
            try:
                response = ipc.recv().decode()
            except:
                print("Environment has been disconnected")
                raise ValueError('Disconnected')
            print("Received {}".format(response))

            time.sleep(DELAY)
        except:
            print("Shutting down agent controller")
            exit()
