#!/usr/bin/env python3
#---------------------------------
# AASMA - Environment
# File : run.py
#
# @ start date          22 04 2020
# @ last update         22 04 2020
#---------------------------------
from aasma.env import env

if __name__ == '__main__':
    try:
        env.host()
    except:
        raise ValueError('Environment already exists')
