#!/usr/bin/env python3
#---------------------------------
# AASMA - Environment
# File : run.py
#
# @ start date          22 04 2020
# @ last update         24 04 2020
#---------------------------------
import os
from aasma.env import env

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    try:
        env = env.Environment(os.path.join(
            os.path.realpath(__file__)[:-6],
            './config.json'
        )).run()
    except:
        raise ValueError('Environment couldn\'t be deployed')
