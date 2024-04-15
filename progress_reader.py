import os
import pyautogui as ag
from PIL import Image
import pygetwindow as gw
import time
import mss
import numpy as np

#SETTINGS:
#Resolution: 1280x720 
#Framerate: 60fps
#Progress bar: on

#exact values for location of progress bar might need to be tweaked depending on resolution, I used 

#progress only tracked if window is active

FRAMERATE = 60.0

def getProgress(mss_img):
    #Sampling the pixels of the bar
    samples = range(408, 867)
    height = 15
    green_count = 0
    for x in samples:
        pixel = mss_img.pixel(x,height)
        if(pixel[1] > 160 and pixel[0] < pixel[1] and pixel[2] < pixel[1]):
            green_count += 1
    return green_count/len(samples)
    

GDwindow = gw.getWindowsWithTitle("Geometry Dash")[0]

GDwindow.activate()
screenWidth, screenHeight = GDwindow.size

with mss.mss() as sct:
    while(1):
        monitor = {"top": GDwindow.top + 33, "left": GDwindow.left + 10, "width": screenWidth-20, "height": screenHeight - 38}
        screengrab = sct.grab(monitor)
        screen_data = np.array(screengrab)
        
        progress = getProgress(screengrab)
        print(progress)
        if(progress > .9):
            Image.fromarray(screen_data).show()
            exit()
        time.sleep(1/FRAMERATE)
    

#ag.click(x=screenWidth//2, y=screenHeight//2)