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
# function: snapshot
#---------------------------------
def snapshot():
    env_window = gw.getWindowsWithTitle('AASMA Environment')
    if len(env_window) != 1:
        raise ValueError('Environment doesn\'t exist or isn\'t unique')

    env_window = env_window[0]
    # Focus window
    env_window.activate()
    # Move window to standard position
    env_window.moveTo(0, 0)

    env_width, env_height = env_window.size

    screen = np.array(ImageGrab.grab(bbox=(0, 0, env_width, env_height)))
    return screen

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    last_time = time.time()
    while True:
        img = cv2.cvtColor(snapshot(), cv2.COLOR_BGR2RGB)

        print("Loop took {} seconds".format(time.time() - last_time))
        last_time = time.time()
        cv2.imshow('window', img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
