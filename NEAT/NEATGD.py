from NEAT.FrameHelper import FrameProcessor
from NEAT.NEATController import NEATController
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