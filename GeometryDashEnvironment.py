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
    
    def __init__(self, top_offset=0, left_offset=0, width_offset=0,
                 height_offset=0, survival_reward=1, death_penalty=-100):
        super().__init__()
        self.keyboard = Controller()
        self.g_window = gw.getWindowsWithTitle("Geometry Dash")[0]
        self.frame_processor = FrameProcessor("Geometry Dash", top_offset, 
                                              left_offset, width_offset, height_offset)
        self.win_width, self.win_height = self.frame_processor.get_frame_shape()
        # The observation space is the numpy array of pixel values from screenshots
        # of Geometry Dash. The shape is dimensions of the screenshots used.
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(self.win_height, self.win_width), 
                                            dtype=np.uint8)
        # The two actions are 0 - idle, and 1 - Jump
        self.action_space = spaces.Discrete(2)
        self.observation = np.empty([self.win_height, self.win_width],dtype=np.uint8)
        self.reward = 0
        self.survival_reward = survival_reward
        self.death_penalty = death_penalty
    
    def _get_obs(self):
        return self.observation
    
    def reset(self):
        # Restart Geometry Dash
        self.g_window.activate()
        # Presses space to restart level
        self.press_space()
        # Sleeps for 0.5 serconds to allow death screen to fully dissappear. 
        # Otherwise, environment thinks agent died multiple times in one iteration.
        time.sleep(0.5)                    
        self.observation = self.frame_processor.get_frame()
        self.reward = 0
        return self.observation
        
    def compute_reward(self, prev_obs, observation):
        reward = self.reward
        terminated = is_dead(prev_obs, observation)
        if terminated:
            reward += self.death_penalty
        else:
            reward += self.survival_reward
        return reward, terminated
    
    def step(self, action):
        observation = self._get_obs()
        if action == 1:
            self.press_space()
        self.observation = self.frame_processor.get_frame()
        self.reward, terminated = self.compute_reward(observation,self.observation)
        return self.observation, self.reward, terminated
    
    def _render_frame(self):
        title = "Geometry Dash Environment Viewer"
        cv2.imshow(title, self.observation)
        if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
    
    def render(self):
        return self._render_frame()
        
        
    def press_space(self):
        self.keyboard.press(Key.space)
        self.keyboard.release(Key.space)
            

