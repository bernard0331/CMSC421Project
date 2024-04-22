from FrameHelper import FrameProcessor
from EnvironmentHelper import is_dead
import cv2
from GeometryDashEnvironment import GDashEnv



#Constants for the window titles of Geometry Dash and The Impossible Game for easy switching
GD = "Geometry Dash"
IG = "The Impossible Game"

IS_DEAD_PATH = ".venv\\images\\isDead.jpg"
NOT_DEAD_PATH = ".venv\\images\\notDead.jpg"

# Change this between GD or IG to select the game
GAME = GD

frame_processor = FrameProcessor(GAME,60,100,240,160)

# The amount of fps we can achieve with the current 
# hardware and preprocessing.
# print("FPS: ", frame_processor.calculate_fps())

# Displays the frames of geometry dash for 30 seconds.
# Feel free to change the duration.
frame_processor.display_frames(30)
#frame_processor.save_sct("isDead.jpg")
#frame_processor.save_sct("notDead.jpg")
# frame_processor.update_terminal_image()