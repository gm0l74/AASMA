#!/usr/bin/env python3
#---------------------------------
# AASMA - Agent
# File : agent.py
#
# @ start date          22 04 2020
# @ last update         02 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import sys, zmq
import grabber # custom

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


    # TODO
    #  Socket to talk to environment communicator
    print("Connecting to communicator...")
    socket = zmq.Context().socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    #  Do 10 requests, waiting each time for a response
    for request in range(10):
        print("Sending request %s …" % request)
        socket.send(b"Hello")

        #  Get the reply.
        message = socket.recv()
        print("Received reply %s [ %s ]" % (request, message))
