#!/bin/bash
#---------------------------------
# AASMA
# File : builer.sh
#
# @ start date          22 04 2020
# @ last update         22 04 2020
#---------------------------------
# Error handle these arguments
if [ -z "$1" ] ; then
    echo "Error: OSFLAG is required"
fi

# Install requirements and set environment variables
#if [ -z ${IS_BUILT+x} ]; then
#  pip3 install -r requirements.txt --user
#fi

# Create and deploy the aasma module locally
python3 setup.py install --user
chmod +x aasma.sh aasma.py

# Give executable priveleges to all python scripts
find aasma/ -name '*.py' -type f ! -name __init__.py -exec chmod +x {} \;

# Insert the alias
if [ -z ${IS_BUILT+x} ]; then
  if [ "$1" = "WIN32" ]; then
    doskey aasma=$(pwd)/aasma.sh
  else
    echo "alias aasma=\"bash $(pwd)/aasma.sh\"" >> ${HOME}/.bashrc
  	source ${HOME}/.bashrc
  fi
fi
