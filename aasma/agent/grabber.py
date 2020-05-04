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
    screen = np.array(ImageGrab.grab(bbox=(0,40,600,630)))
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
