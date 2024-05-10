from FrameHelper import FrameProcessor
from NEATController import NEATController
import os
import skimage
from skimage import exposure
from skimage import io
import cv2
import numpy as np
from PIL import Image
import time

#Constants for the window titles of Geometry Dash and The Impossible Game for easy switching
GD = "Geometry Dash"
IG = "The Impossible Game"

# Change this between GD or IG to select the game
GAME = GD

frame_processor = FrameProcessor(GAME,60,170,300,80)

# The amount of fps we can achieve with the current     
# hardware and preprocessing.
#print("FPS: ", frame_processor.calculate_fps()  )

#Prints model image
#time.sleep(5)
#img = frame_processor.get_frame()
#img = cv2.merge((img,img,img))
#img = skimage.transform.resize(img, (46, 22), anti_aliasing=True)
#img = exposure.adjust_gamma(img, gamma=.25, gain=1)      
#img = Image.fromarray((img * 255).astype(np.uint8))
#img.show()

# Displays the frames of geometry dash for 30 seconds.
# Feel free to change the duration.
#print(frame_processor.display_frames(30))


#img = frame_processor.get_raw_frame(35,180,330,100)
#img = Image.fromarray(img)
#img.show()

#Runs NEAT
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')

saveload_path = os.path.join(local_dir, 'cache.pkl')

print("Starting NEAT")
nc = NEATController(config_path, frame_processor)
nc.evolve(0)
nc.load_checkpoint("Final-CP")
nc.evolve(1000)

#print("Winner decided, saving winner")
#nc.save_model(saveload_path)
#print("Saved, playing final model")
#progress = nc.eval()
#print(progress)       
#nc.show_model()