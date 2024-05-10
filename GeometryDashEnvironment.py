import gymnasium as gym
from gymnasium import spaces
import numpy as np
from FrameHelper import FrameProcessor
import EnvironmentHelper
from EnvironmentHelper import is_dead_progress, is_dead_progress_space, SpaceKeyHandler
import pygetwindow as gw
import cv2
from pynput.keyboard import Key, Controller
import time


class GDashEnv(gym.Env):
    """A custom gymnasium environment designed for the game Geometry Dash. 
    Requires that Geometry Dash be running when being used.
    """
    
    def __init__(self, top_offset=40, left_offset=80, width_offset=160,
                 height_offset=120, win_width=260, win_height=220, survival_reward=1, death_penalty=-100, down_scaling=False, 
                 scale_width_factor=1, scale_height_factor=1, scale_width_offset=60, 
                 scale_height_offset=140, jump_penalty=0.1, goal_reward=10, space=False):
        """Constructor for a GDashEnv object. Arguments determine the size of the screenshots
        of the game window and the reward values being used.

        Args:
            top_offset (int, optional): Defines top coordinate of screenshot of game used 
                relative to the top of the game window. Defaults to 80.
            left_offset (int, optional): Defines the left coordinate of screenshot of game
                used relative to left edge of game window. Defaults to 100.
            width_offset (int, optional): The amount of pixels subtracted from the width
                of the game window. Defaults to 240.
            height_offset (int, optional): The amount of pixels subtracted from the height
                of the game window. Defaults to 160.
            survival_reward (int, optional): The reward for surving one step. Defaults to 1.
                death_penalty (int, optional): The penalty for dying. Defaults to -100.
            down_scaling (bool, optional): Decides if downscaling is used when grabbing frames.
                Defaults to False.
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
                                              scale_height_factor, scale_width_offset, scale_height_offset, 
                                              win_width=win_width, win_height=win_height)
        self.win_width, self.win_height = self.frame_processor.get_frame_shape()
        # The observation space is the numpy array of pixel values from screenshots
        # of Geometry Dash. The shape is dimensions of the screenshots used.
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(self.win_height, self.win_width), 
                                            dtype=np.uint8)
        # The two actions are 0 - idle, and 1 - Jump
        self.action_space = spaces.Discrete(2)
        # reset() method will initialize self.observation with a screenshot of the game.
        self.observation = np.empty([self.win_height, self.win_width],dtype=np.uint8)
        # self.reward keeps track of the current total of the current iteration.
        self.total_reward = 0
        self.reward = 0
        self.survival_reward = survival_reward
        self.death_penalty = death_penalty
        self.cur_jumps = 0
        self.jump_penalty = jump_penalty
        self.goal_reward = goal_reward
        self.JUMP = 1
        self.IDLE = 0
        self.progress = 0.0
        self.space = space
        self.space_handler = SpaceKeyHandler()
    
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
        # time.sleep(0.1)
        # Makes sure Geometry Dash window is in focus.
        self.g_window.activate()
        # Presses space to start or restart level.
        self.release_space()
        self.press_space()
        # Sleeps for 0.5 seconds to allow death screen to fully dissappear. 
        # Otherwise, environment thinks agent died multiple times in one iteration.
        # time.sleep(0.3)                    
        self.observation = self.frame_processor.get_frame()
        self.reward = 0
        self.total_reward = 0
        self.progress = 0.0
        self.cur_jumps = 0        
        return self.observation
        
    def compute_reward(self,action):
        """Computes the reward for the current step.

        Args:
            prev_obs (Matlike): The screenshot of the previous game frame.
            observation (Matlike): The screenshot of the current game frame.

        Returns:
            (int, bool): The updated current reward total and whether the game 
                is terminated.
        """
        jump_pen = 0
        if action == 1:
            jump_pen = self.jump_penalty
        reward = 0.0
        raw_img = self.frame_processor.get_raw_frame()
        
        if self.space:
            terminated, progress = is_dead_progress_space(raw_img, self.progress)
        else:
            terminated, progress = is_dead_progress(raw_img, self.progress)
        
        if progress > 99:
            terminated = True
            reward = self.goal_reward
        elif terminated:
            reward = self.survival_reward - jump_pen + self.death_penalty
            #reward = self.survival_reward * self.progress + self.cur_jump_penalty + self.death_penalty
        else:
            reward = self.survival_reward - jump_pen
            #reward = self.survival_reward * progress + self.cur_jump_penalty
        self.progress = progress
        self.reward = reward
        self.total_reward += reward
        return reward, terminated, progress
    
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
        
        if self.space:
            if action == 1:
                self.press_space_ship()
                self.cur_jumps += 1
            else:
                self.release_space()
        if action == 1:     # Agent jumps.
            self.press_space()
            self.cur_jumps += 1
        # Gets the next frame.
        self.observation = self.frame_processor.get_frame()
        # Computes new reward after the action is performed.
        reward, terminated, progress = self.compute_reward(action)
        return self.observation, reward, terminated, progress
    
    def render(self):
        """Displays the current frame of the game.
        """
        title = "Geometry Dash Environment Viewer"
        img = self.frame_processor.prepare_for_nn(self.observation)
        img = np.squeeze(img)
        cv2.imshow(title, img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
        
        
    def press_space(self):
        """Inputs one space bar click. Mainly used to perform a jump in Geometry
        Dash.
        """
        self.keyboard.press(Key.space)
        self.keyboard.release(Key.space)
        
    def press_space_ship(self):
        self.space_handler.press()
    
    def release_space(self):
        self.space_handler.release()
