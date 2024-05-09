from GeometryDashEnvironment import GDashEnv
from NeuralNetwork import NeuralNetwork
import numpy as np
from FrameHelper import FrameProcessor
import time
import os
os.environ["KERAS_BACKEND"] = "torch"
from keras import models as km

### IMPORTANT ###
# To run properly: 
# 1) Launch Geometry Dash.
# 2) Start the level you want to test first.
# 3) Let yourself die once.
# 4) With the death screen present, start this driver.
# Note: The step() method presses spacebar, so if you accidentally click out of
# Geometry Dash while the programming is running it could lead to spam spacing.
# I belive ctrl + c is the shortcut to terminate the program (must have your terminal
# in focus).

FPS = 100
MAX_FRAME_TIME = 1/FPS

living_reward = 3
death_penalty = -10
gamma = 0.98
exploration_rate = 1.0
exploration_decay = 0.99
lr = 0.0005
batch_size = 32
max_exp = 100000
n = 20
jump_pen = 0.025         
win_width, win_height = 260,220
top,left,width_off,height_off = 96,104,122,138                        
game = GDashEnv(top_offset=top, left_offset=left, width_offset=width_off, height_offset=height_off, 
                win_width=win_width, win_height=win_height, survival_reward=living_reward, 
                death_penalty=death_penalty, down_scaling=False, scale_height_factor=1, 
                scale_width_factor=1, scale_width_offset=180, scale_height_offset=208,
                jump_penalty = jump_pen)

# Initialize FrameProcessor with the window title of the game
frame_processor = game.frame_processor  # Ensure this line is correctly placed before it's used

# Initializing the neural network with the game environment
nn = NeuralNetwork(gamma=gamma, lr=lr, batch_size=batch_size, max_experience=max_exp,
                   exp=exploration_rate, exp_decay=exploration_decay,env=game,n=n)
# nn.neural = km.load_model(".venv\\Models\\michaelScott.keras")

run = 0
iterations = 100 # Number of iterations you want to run
# NDArray of length iterations used to track the progress of each run
training_results = np.empty([iterations]) 
start_time = time.time()
fps = 0
while time.time() - start_time < 30:
    if nn.frame_count % nn.target_frames == 0:
        nn.update_target()
    observation = game.reset()
    observation = frame_processor.prepare_for_nn(observation)  # Use centralized preprocessing
    terminated = False
    while not terminated and time.time() - start_time < 30:
        action = nn.predict_action(observation)
        new_obs, reward, terminated, progress = game.step(action)
        # Used to get how much progress was achieved in that iteration
        if progress > 0.0:
            max_progress = progress
        new_obs = frame_processor.prepare_for_nn(new_obs)  # Ensure observation is processed for the next cycle
        nn.save_experience(observation, action,  reward, new_obs ,terminated)
        nn.update_network()
        observation = new_obs
        #game.render()
        nn.frame_count += 1
        fps += 1
    if nn.exp > nn.exp_min:
        nn.exp *= nn.exp_decay
    training_results[run] = max_progress
    run += 1
    print("Run #", run, " Reward: ", reward, " Progress: %", max_progress)

print("FPS: ", fps/30)

# nn.neural.save(".venv\\Models\\michaelScott.keras")
print("Average first 10 iterations: ", np.mean(training_results[:10]))
print("")
print("Average last 10 iterations: ", np.mean(training_results[-10:]))
print("Current Exploration Rate: ", nn.exp)