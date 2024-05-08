import numpy as np
import cv2
import pygetwindow as gw
import mss
import time
from EnvironmentHelper import is_dead_progress, getProgress

class FrameProcessor:
    """A class designed to grab screenshots of a desired window, process those screenshots, and 
    perform desired actions with them.
    """
    
    def __init__(self, win_title, top_offset=0, left_offset=0, width_offset=0,
                 height_offset=0, down_scaling=False, scale_width_factor=1,
                 scale_height_factor=1, scale_width_offset=60, scale_height_offset=100,
                 win_width=260, win_height=220):
        """Initializes a FrameProcessor object designed to grab screenshots of a 
        specific window and perform certain actions with it.

        Args:
            win_title (str): The title of the target window.
            top_offset (int, optional): Offset for the top border of the screenshot. Defaults to 0.
            left_offset (int, optional): Offset for the left border of the screenshot. Defaults to 0.
            width_offset (int, optional): Offset for the width of the screenshot. Defaults to 0.
            height_offset (int, optional): Offset for the height of the screenshot. Defaults to 0.
            down_scaling (bool, optional): Decides if frame is downscaled. Defaults to False.
            scale_width_factor (float, optional): Sets the width factor for downscaling. Defaults to 1.
            scale_height_factor (float, optional): Sets the height factor for downscaling. Defaults to 1.
            scale_width_offset (int, optional) The amount of pixels taken off the width in downscaling. 
                Defaults to 60.
            scale_height_offset (int, optional) The amount of pixels taken off the height in downscaling. 
                Defaults to 100.
        """
        self.win_title = win_title
        self.g_window = gw.getWindowsWithTitle(win_title)[0]
        self.g_window.size = (win_width, win_height)
        self.update_monitor_dimensions(top_offset, left_offset, width_offset, height_offset)
        self.sct = mss.mss()
        self.down_scaling = down_scaling
        self.scale_width_factor = scale_width_factor
        self.scale_height_factor = scale_height_factor
        self.scale_width_offset = scale_width_offset
        self.scale_height_offset = scale_height_offset
    
    def update_monitor_dimensions(self, top_offset, left_offset, width_offset, height_offset):
        """Updates the monitor capture dimensions based on the current position and size of the window."""
        self.g_window = gw.getWindowsWithTitle(self.win_title)[0]  # Re-acquire the window
        self.mon = {
            "top": self.g_window.top + top_offset,
            "left": self.g_window.left + left_offset,
            "width": max(self.g_window.width - width_offset, 1),  # ensure non-negative width
            "height": max(self.g_window.height - height_offset, 1)  # ensure non-negative height
        }   
    
    def get_frame_shape(self):
        if self.down_scaling:
            return ((self.mon["height"] - self.scale_height_offset) * self.scale_height_factor,
                    (self.mon["width"] - self.scale_width_offset) * self.scale_width_factor)
        return (self.mon["height"], self.mon["width"])
    
    def get_frame(self):
        """Grabs a screenshot of the selected window.

        Returns:
            NDArray: Contains pixel information of the screenshot
        """
        img = np.asarray(self.sct.grab(self.mon))
            
        return img
    
    def get_raw_frame(self, top_offset=33, left_offset=77, width_offset=154,
                 height_offset=219):
        """Grabs an unprocessed screenshot of the selected window 
        and returns it as a NDArray of pixel values. Default values
        grab a screenshot of the progress bar.
        
        Returns:
            Matlike: NDArray containing the pixels of a screenshot
        """
        mon = {"top": self.g_window.top + top_offset, "left": self.g_window.left + left_offset, 
                    "width": self.g_window.width - width_offset, "height": self.g_window.height - height_offset}
        return np.asarray(self.sct.grab(mon))
    
    def prepare_for_nn(self, img):
        """Prepares the image for neural network processing by converting to grayscale, resizing, and adding a channel dimension."""
        if img.shape[-1] == 3 or img.shape[-1] == 4:  # Check if image has color channels
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.down_scaling:
            img = cv2.resize(img, (self.get_frame_shape()[1],self.get_frame_shape()[0]), interpolation=cv2.INTER_AREA)
        # Reducing noise, trying to get only the important lines and shapes (cube, platform, obstacles)
        # Higher thresholds seems to reduce noise (unecessary lines and shapes)
        img = cv2.Canny(img, 150, 250)
        img = cv2.normalize(img, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = np.expand_dims(img, axis=-1)  # Add channel dimension for neural network compatibility
        # img = np.expand_dims(img, axis=0)
        return img
    
    def calculate_fps(self, with_display=False) -> int:
        """Calculates how many frames can be produced per second and returns it.
        Note: Does not take into account time added by  extra processes such as 
        neural network training. Real fps likely noticably lower in practice."""
        fps = 0
        last_time = time.time()
        
        while time.time() - last_time < 5:
            img = self.get_frame()
            img = self.prepare_for_nn(img)
            if with_display:
                title = self.win_title + " Viewer"
                cv2.imshow(title,img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
            fps += 1
        
        return fps/5
    
    def display_frames(self, duration, by_frames=False):
        """Displays the screenshots of the selected window.

        Args:
            duration (int): The length of time in seconds that the frames are displayed (or the 
                number of frames displayed if by_frames=True).
            by_frames (bool, optional): If set to True, duration is the number of frames to 
                be displayed. Defaults to False.
        """
        title = self.win_title + " Viewer"
        iterations = duration if by_frames else duration * self.calculate_fps()
        while iterations > 0:
            img = self.get_frame()
            img = self.prepare_for_nn(img)
            img = np.squeeze(img)
            cv2.imshow(title, img)
            iterations -= 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
    
    def display_is_dead_frames(self,duration,top_offset=33, left_offset=77, 
                               width_offset=154, height_offset=219, by_frames=False):
        """Method used to test is_dead_progress and display progress bar.

        Args:
            duration (int): The length of time in seconds that the frames are displayed (or the 
                number of frames displayed if by_frames=True).
            by_frames (bool, optional): If set to True, duration is the number of frames to 
                be displayed. Defaults to False.
        """
        title = self.win_title + " Viewer"
        iterations = duration if by_frames else duration * self.calculate_fps()
        prev_progress = 0.0
        while iterations > 0:
            img = self.get_raw_frame(top_offset,left_offset,width_offset,height_offset)
            # img = self.prepare_for_nn(img)
            # img = np.squeeze(img)
            terminated,progress = is_dead_progress(img,prev_progress)
            # print(progress)
            if terminated:
                print("DEAD")
            prev_progress = progress
            cv2.imshow(title, img)
            iterations -= 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
    
    def save_sct(self, sct_name, path=".venv\\images\\"):
        img = self.get_frame()
        name = path + sct_name
        
        np.save(name, img)
        