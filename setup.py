#!/usr/bin/env python3
#---------------------------------
# AASMA
# File : setup.py
#
# @ start date          21 05 2020
# @ last update         22 05 2020
#---------------------------------

#---------------------------------
# Usage
#---------------------------------
# To install package locally
# python setup.py install --user

#---------------------------------
# Imports
#---------------------------------
from setuptools import setup, find_packages

#---------------------------------
# Execute
#---------------------------------
setup(
    name='aasma',
    version='1.1',
    packages=find_packages()
)
