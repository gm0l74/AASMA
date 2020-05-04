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

import aasma.agent.grabber

import aasma.agent.models.drn
import aasma.agent.models.randomness

#---------------------------------
# Constants
#---------------------------------
# Environment engine configuration
FPS = 15 # frames per second (in Hz)

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    # Error handle script arguments
    args = sys.argv
    if len(args) != 1:
        raise ValueError('Invalid number of arguments')

    # Error handle agent type
    agent_type = args
    if agent_type not in ('randomness', 'drl'):
        raise ValueError('Invalid agent type')

    #  Connect to environment communicator
    print("Connecting to communicator...")
    ipc = zmq.Context().socket(zmq.REQ)
    ipc.connect("tcp://localhost:5555")

    # Send spawn notification
    print("Sending spawn request")
    ipc.send(b"create")

    # Error handle the response
    agent_id = socket.recv()
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
            agent.perceive(grabber.snapshot())

            # Reason on some action
            action = agent.make_action()

            # Send the selected action...
            print("Sending selected action ({})".format(action))
            query = "move,{},{}".format(agent_id, action)
            ipc.send(query)

            #  ... and get the response
            response = socket.recv().decode()
            print("Received {}".format(response))

            # Error handle it
            if response != action:
                raise ValueError("Action was not understood by ipc")

            time.sleep(1/FPS)
        except:
            print("Shutting down agent controller")
            exit()
