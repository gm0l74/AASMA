#---------------------------------
# AASMA
# File : Makefile
#
# @ start date          22 04 2020
# @ last update         22 04 2020
#---------------------------------

# Detect operating system
OSFLAG :=
ifeq ($(OS),Windows_NT)
	OSFLAG += WIN32
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		OSFLAG += LINUX
	endif
	ifeq ($(UNAME_S),Darwin)
		OSFLAG += OSX
	endif
endif

build:
	chmod +x builder.sh
	bash builder.sh ${OSFLAG}

uninstall:
	-rm -r aasma.egg-info dist build

reset: uninstall build
