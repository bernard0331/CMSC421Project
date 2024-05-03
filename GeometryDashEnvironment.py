import gymnasium as gym
from gymnasium import spaces
import numpy as np
from FrameHelper import FrameProcessor
import EnvironmentHelper
from EnvironmentHelper import is_dead
import pygetwindow as gw
import cv2
from pynput.keyboard import Key, Controller
import time

class GDashEnv(gym.Env):
    """A custom gymnasium environment designed for the game Geometry Dash. 
    Requires that Geometry Dash be running when being used.
    """
    JUMP = 1
    IDLE = 0
    
    def __init__(self, top_offset=30, left_offset=10, width_offset=20,
                 height_offset=40, survival_reward=1, death_penalty=-100, 
                 is_dead_threshold=0.91, down_scaling=True, scale_width_factor=1,
                 scale_height_factor=1, scale_width_offset=60, scale_height_offset=140):
        """Constructor for a GDashEnv object. Arguments determine the size of the screenshots
        of the game window and the reward values being used.

        Args:
            top_offset (int, optional): Defines top coordinate of screenshot of game used 
                relative to the top of the game window. Defaults to 0.
            left_offset (int, optional): Defines the left coordinate of screenshot of game
                used relative to left edge of game window. Defaults to 0.
            width_offset (int, optional): The amount of pixels subtracted from the width
                of the game window. Defaults to 0.
            height_offset (int, optional): The amount of pixels subtracted from the height
                of the game window. Defaults to 0.
            survival_reward (int, optional): The reward for surving one step. Defaults to 1.
                death_penalty (int, optional): The penalty for dying. Defaults to -100.
            is_dead_threshold (float, optional): The threshold for is_dead() method used to 
                determine if the agent has died, a value between 0 and 1.0. Defaults to 0.91.
            down_scaling (bool, optional): Decides if downscaling is used when grabbing frames.
                Defaults to True.
            scale_width_factor (float, optional) The width scale factor for downscaling. 
                Defaults to 1.
            scale_height_factor (float, optional) The height scale factor for downscaling. 
                Defaults to 1.
            scale_width_offset (int, optional) The amount of pixels taken off the width in downscaling. 
                Defaults to 60.
            scale_height_offset (int, optional) The amount of pixels taken off the height in downscaling. 
                Defaults to 100.
        """
        super().__init__()
        self.keyboard = Controller()
        self.g_window = gw.getWindowsWithTitle("Geometry Dash")[0]
        # FrameProcessor is used to get simplified screenshots of the game in real-time.
        self.frame_processor = FrameProcessor("Geometry Dash", top_offset, 
                                              left_offset, width_offset, height_offset,
                                              down_scaling, scale_width_factor,
                                              scale_height_factor, scale_width_offset, scale_height_offset)
        self.win_width, self.win_height = self.frame_processor.get_frame_shape()
        # The observation space is the numpy array of pixel values from screenshots
        # of Geometry Dash. The shape is dimensions of the screenshots used.
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(self.win_height, self.win_width), 
                                            dtype=np.uint8)
        # The two actions are 0 - idle, and 1 - Jump
        self.action_space = spaces.Discrete(4)
        # reset() method will initialize self.observation with a screenshot of the game.
        self.observation = np.empty([self.win_height, self.win_width],dtype=np.uint8)
        # self.reward keeps track of the current total of the current iteration.
        self.reward = 0
        self.survival_reward = survival_reward
        self.death_penalty = death_penalty
        self.is_dead_threshold = is_dead_threshold
    
    def _get_obs(self):
        """Basic getter for the current observation (current frame of the game)."""
        return self.observation
    
    def reset(self):
        """Resets the environment by starting or restarting the current Geoometry Dash
        level. Then sets the current observation to a new screenshot of the game and 
        resets the reward to 0.

        Returns:
            Matlike: The current observation (current frame of the game).
        """
        time.sleep(0.1)
        # Makes sure Geometry Dash window is in focus.
        self.g_window.activate()
        # Presses space to start or restart level.
        self.press_space()
        # Sleeps for 0.5 serconds to allow death screen to fully dissappear. 
        # Otherwise, environment thinks agent died multiple times in one iteration.
        time.sleep(0.5)                    
        self.observation = self.frame_processor.get_frame()
        self.reward = 0
        return self.observation
        
    def compute_reward(self, prev_obs, observation):
        """Computes the reward for the current step.

        Args:
            prev_obs (Matlike): The screenshot of the previous game frame.
            observation (Matlike): The screenshot of the current game frame.

        Returns:
            (int, bool): The updated current reward total and whether the game 
                is terminated.
        """
        reward = self.reward
        terminated = is_dead(prev_obs, observation, self.is_dead_threshold)
        if terminated:
            reward += self.death_penalty
        else:
            reward += self.survival_reward
        return reward, terminated
    
    def step(self, action):
        """Conducts one step by performing the provided action and interpretting
        the outcome of that action.

        Args:
            action (int): 1 for jump, anything else for idling.

        Returns:
            (Matlike, int, bool): The current observation (screenshot of the current
                frame) post-action, the current total reward, and whether the game is 
                terminated.
        """
        
        observation = self._get_obs()
        if action == 1:     # Agent jumps.
            self.press_space()
        # Gets the next frame.
        self.observation = self.frame_processor.get_frame()
        # Computes new reward after the action is performed.
        self.reward, terminated = self.compute_reward(observation,self.observation)
        return self.observation, self.reward, terminated
    
    def render(self):
        """Displays the current frame of the game.
        """
        title = "Geometry Dash Environment Viewer"
        cv2.imshow(title, self.observation)
        if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
        
        
    def press_space(self):
        """Inputs one space bar click. Mainly used to perform a jump in Geometry
        Dash.
        """
        self.keyboard.press(Key.space)
        self.keyboard.release(Key.space)

