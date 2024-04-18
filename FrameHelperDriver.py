from FrameHelper import FrameProcessor

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
print(frame_processor.display_frames(30))