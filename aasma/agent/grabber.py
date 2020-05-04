#!/usr/bin/env python3
#---------------------------------
# AASMA - Agent
# File : grabber.py
#
# @ start date          22 04 2020
# @ last update         04 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import numpy as np
from PIL import ImageGrab
import cv2
import time

#---------------------------------
# function: snapshot
#---------------------------------
def snapshot():
    screen = np.array(ImageGrab.grab(bbox=(0,40,600,600))) # TODO
    return cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

#---------------------------------
# Execute
#---------------------------------
if __name__ == '__main__':
    last_time = time.time()
    while True:
        img = snapshot()

        print("Loop took {} seconds".format(time.time() - last_time))
        cv2.imshow('window', img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
