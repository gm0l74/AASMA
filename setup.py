#!/usr/bin/env python3
#---------------------------------
# AASMA
# File : setup.py
#
# @ start date          22 04 2020
# @ last update         28 04 2020
#---------------------------------
from setuptools import setup, find_packages

setup(
    name='aasma',
    version='1.1',
    data_files=[
        'aasma/env/sprites/mountain.png',
        'aasma/env/sprites/fire.png',
        'aasma/env/sprites/drone.png'
    ],
    packages=find_packages()
)
