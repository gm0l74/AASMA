#!/bin/bash
#---------------------------------
# AASMA
# File : aasma.sh
#
# @ author              gmoita
#
# @ start date          22 04 2020
# @ last update         22 04 2020
#---------------------------------

# Root directory of aasma
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#---------------------------------
# Execute
#---------------------------------
python3 $ROOT/aasma.py "${@:1}"
