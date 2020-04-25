#!/usr/bin/env python3
#---------------------------------
# AASMA
# File : setup.py
#
# @ start date          22 04 2020
# @ last update         25 04 2020
#---------------------------------
from setuptools import setup, find_packages
# TODO
setup(
    name='aasma',
    version='1.0',
    data_files=[
        'aasma/env/sprites/mountain.png',
        'aasma/env/sprites/fire.png'
    ],
    packages=find_packages()
)
