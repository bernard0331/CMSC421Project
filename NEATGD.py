from FrameHelper import FrameProcessor
from NEATHelper import NEATController
import os

#from PIL import Image

#Constants for the window titles of Geometry Dash and The Impossible Game for easy switching
GD = "Geometry Dash"
IG = "The Impossible Game"

# Change this between GD or IG to select the game
GAME = GD

frame_processor = FrameProcessor(GAME,60,100,240,160)

# The amount of fps we can achieve with the current     
# hardware and preprocessing.
print("FPS: ", frame_processor.calculate_fps())

# Displays the frames of geometry dash for 30 seconds.
# Feel free to change the duration.
#print(frame_processor.display_frames(30))



#img = Image.fromarray(frame_processor.get_raw_frame(35,180,330,500))
#img.show()

#Runs NEAT
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')

print("Starting NEAT")
nc = NEATController(config_path, frame_processor)
nc.evolve(1)
print("Winner decided, showing final model")
nc.eval()
nc.show_model()