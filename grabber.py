#!/usr/bin/env python3
#---------------------------------
# AASMA - Agent
# File : grabber.py
#
# @ start date          22 04 2020
# @ last update         05 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
from PIL import ImageGrab
import pygetwindow as gw
import numpy as np
import cv2, time

#---------------------------------
# Foreplay
#---------------------------------
env_win = gw.getWindowsWithTitle('AASMA Environment')
if len(env_win) != 1:
    raise ValueError("Environment doesn't exist or isn't unique")

env_win = env_win[0]
# Focus window
env_win.activate()
# Move window to standard position
env_win.moveTo(0, 0)

ENV_WIDTH, ENV_HEIGHT = env_win.size

#---------------------------------
# function: snapshot
#---------------------------------
def snapshot():
    return np.array(ImageGrab.grab(
        bbox=(8, 31, ENV_WIDTH - 8, ENV_HEIGHT - 8)
    ))

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    last_time = time.time()
    while True:
        img = cv2.cvtColor(snapshot(), cv2.COLOR_BGR2RGB)

        print("{:.5f} seconds".format(time.time() - last_time))
        last_time = time.time()
        cv2.imshow('DRL Input Feed', img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
