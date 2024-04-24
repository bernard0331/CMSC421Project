import numpy as np
import cv2
import pygetwindow as gw
import mss
import time
from EnvironmentHelper import is_dead

class FrameProcessor:
    """A class designed to grab screenshots of a desired window, process those screenshots, and 
    perform desired actions with them.
    """
    
    def __init__(self, win_title, top_offset=0, left_offset=0, width_offset=0,
                 height_offset=0):
        """Initializes a FrameProcessor object designed to grab screenshots of a 
        specific window and perform certain actions with it.

        Args:
            win_title (str): the title of the target window.
            top_offset (int, optional): offset for the top border of the screenshot. Defaults to 0.
            left_offset (int, optional): offset for the left border of the screenshot. Defaults to 0.
            width_offset (int, optional): offset for the width of the screenshot. Defaults to 0.
            height_offset (int, optional): offset for the height of the screenshot. Defaults to 0.
        """
        self.win_title = win_title
        self.g_window = gw.getWindowsWithTitle(win_title)[0]
        self.win_width, self.win_height = self.g_window.size
        self.mon = {"top": self.g_window.top + top_offset, "left": self.g_window.left + left_offset, 
                    "width": self.win_width - width_offset, "height": self.win_height - height_offset}
        self.sct = mss.mss()
        
    def get_frame_shape(self):
        return (self.mon.get("width"), self.mon.get("height"))
        
    
    def get_frame(self):
        """Grabs a screenshot of the selected window, simplifies the image with
        processing and returns it as a NDArray of pixel values.

        Returns:
            Matlike: NDArray containing the simplified pixels of a screenshot
        """
        img = np.asarray(self.sct.grab(self.mon))
        # Converting image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Reducing noise, trying to get only the important lines and shapes (cube, platform, obstacles)
        # Higher thresholds seems to reduce noise (unecessary lines and shapes)
        img = cv2.Canny(img, 300, 400)
        
        return img
    
    def get_raw_frame(self, top_offset=0, left_offset=0, width_offset=0,
                 height_offset=0):
        """Grabs an unprocessed screenshot of the selected window 
        and returns it as a NDArray of pixel values.
        Returns:
            Matlike: NDArray containing the pixels of a screenshot
        """
        mon = {"top": self.g_window.top + top_offset, "left": self.g_window.left + left_offset, 
                    "width": self.win_width - width_offset, "height": self.win_height - height_offset}
        return np.asarray(self.sct.grab(mon))

    def calculate_fps(self) -> int:
        """Calculates how many frames can be produced per second and returns it"""
        fps = 0
        last_time = time.time()
        
        while time.time() - last_time < 1:
            self.get_frame()
            fps += 1
        
        return fps
    
    def display_frames(self, duration, by_frames=False):
        """Displays the screenshots of the selected window.

        Args:
            duration (int): The length of time in seconds that the frames are displayed (or the 
                number of frames displayed if by_frames=True).
            by_frames (bool, optional): If set to True, duration is the number of frames to 
                be displayed. Defaults to False.
        """
        title = self.win_title + " Viewer"
        iterations = duration
        if not by_frames:
            iterations = duration * self.calculate_fps()
        img = self.get_frame()
        dead = 0
        while iterations > 0:
            new_img = self.get_frame()
            cv2.imshow(title, new_img)
            iterations -= 1
            if is_dead(img, new_img):
                dead += 1
                print("Death: ", dead)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
            img = new_img
    
    # Probably not needed anymore
    def update_terminal_image(self, dead_name="dead", goal_name="goal", path=".venv\\images\\",
                              set_goal=False):
        img = self.get_frame()
        name = path + dead_name
        if set_goal:
            name = path + goal_name
            
        np.save(name, img)
    
    def save_sct(self, sct_name, path=".venv\\images\\"):
        img = self.get_frame()
        name = path + sct_name
        
        np.save(name, img)
        