#---------------------------------
# AASMA
# File : Makefile
#
# @ start date          22 04 2020
# @ last update         22 05 2020
#---------------------------------
build:
	python setup.py install --user

uninstall:
	-rm -r aasma.egg-info dist build

reset: uninstall build
