import numpy as np
import cv2
import pyautogui
from mss import mss
from PIL import Image

def mouse_click(event, x, y, flags, param): 
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button pressed")
    else:
        print("nothing is happening")

screenWidth, screenHeight = pyautogui.size()
bounding_box = {'top': screenHeight/2 + 50, 'left': 0, 'width': screenWidth/2, 'height': screenHeight/2 - 50}

cv2.setMouseCallback('screen', mouse_click)

while True:
    sct_img = mss().grab(bounding_box)
    cv2.imshow('screen', np.array(sct_img))
    #pyautogui.click()
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break