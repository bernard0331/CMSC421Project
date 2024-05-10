from GeometryDashEnvironment import GDashEnv
import EnvironmentHelper as eh
import gymnasium as gym
from NeuralNetwork import NeuralNetwork

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

living_reward = 1
death_penalty = -100
is_dead_threshold = 0.91                                        
game = GDashEnv(80,100,240,160,living_reward,death_penalty,is_dead_threshold, down_scaling=False,
                scale_height_factor=1, scale_width_factor=1, scale_width_offset=60,scale_height_offset=140)
print("Frame shape: ",game.observation_space.shape)

# Increasing action space size so that sample() chooses 1 (jump) less
# game.action_space = gym.spaces.Discrete(6) 

# Initializing the neural network with the game environment
nn = NeuralNetwork(env=game)

run = 0
iterations = 5 # Number of iterations you want to run
print("Frame fps: ",eh.get_environment_fps(game))

for i in range(iterations):
    game.reset()
    terminated = False
    while not terminated:
        # Causes agent to choose action at random (jump or not jump).
        #action = game.action_space.sample()

        # Predicts action based on observation.
        action = nn.predict_action(observation)

        observation, reward, terminated = game.step(action)
        game.render()                         
    run += 1   
    print("Run #", run, " Reward: ", reward)

