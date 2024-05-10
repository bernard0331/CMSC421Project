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

FPS = 120
MAX_FRAME_TIME = 1/120

# Window Parameters
# wind_width should be 260 and win_height should be 220
win_width, win_height = 260,220 #WARNING: changing these will break getProgress and is_dead_progress
top,left,width_off,height_off = 90,80,100,130

# Set true if training on space section, else False
space = True

# Reward parameters
living_reward = 1
death_penalty = -32
goal_reward = 100
jump_pen = 0.5

# Environment initialization                     
game = GDashEnv(top_offset=top, left_offset=left, width_offset=width_off, height_offset=height_off,
                win_width=win_width, win_height=win_height, survival_reward=living_reward, 
                death_penalty=death_penalty, down_scaling=False, scale_height_factor=1, 
                scale_width_factor=1, scale_width_offset=180, scale_height_offset=208,
                jump_penalty=jump_pen, goal_reward=goal_reward, space=space)

# Q-Learning and Neural Network parameters
gamma = 0.98
epsilon = 1.0
eps_decay = 0.985
eps_min = 0.01
lr = 0.001
batch_size = 32
max_exp = 200000
target_frames = 100
n = 20

rand_batch = True                         
double_q = True

# Initializing the neural network with the game environment
nn = NeuralNetwork(gamma=gamma, lr=lr, batch_size=batch_size, max_experience=max_exp,
                   eps=epsilon, eps_decay=eps_decay, env=game, n=n, eps_min=eps_min,
                   target_frames=target_frames, rand_batch=rand_batch, double_q=double_q)

model_file_path= ".venv\\Models\\"
model_name = "michaelScott6.keras"
target_name = "michaelScottTarget6.keras"
log_file_path = ".venv\\TrainingLogs\\michaelScott6.txt"

notes = "Rocket ship part, sesh 2, eps_decay 0.995 -> 0.985, lr 0.0005 -> 0.001, max_exp 100000 -> 200000, \nnew spaceship action behavior - threading"

# set to True if you want to load a saved model (double check name)
# If loading, recommend to adjust epsilon and explore_frames
load = True

iterations = 500 # Number of iterations you want to run

# The amount of frames before epsilon starts decaying
explore_frames = 5000

# set to True to test fps before training
test_fps = False
# Set to True to see render during fps testing - will lower fps
render = False
# set to True to train
train = True

# Testing fps with training steps
if test_fps:
    nn.train_model_FPS(render=render)

# Training the model
if train:
    nn.train_model(iterations, load=load, model_file_path=model_file_path, model_name=model_name,
               target_name=target_name, log_file_path=log_file_path, notes=notes,
               explore_frames=explore_frames)