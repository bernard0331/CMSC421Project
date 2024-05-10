import numpy as np
import cv2
import pygetwindow as gw
import mss
import time


class FrameProcessor:
    """A class designed to grab screenshots of a desired window, process those screenshots, and 
    perform desired actions with them.
    """
    
    def __init__(self, win_title, top_offset=0, left_offset=0, width_offset=0,
                 height_offset=0, down_scaling=False, scale_width_factor=1,
                 scale_height_factor=1, scale_width_offset=60, scale_height_offset=100):
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
        self.win_width, self.win_height = self.g_window.size
        self.mon = {"top": self.g_window.top + top_offset, "left": self.g_window.left + left_offset, 
                    "width": self.win_width - width_offset, "height": self.win_height - height_offset}
        self.sct = mss.mss()
        self.down_scaling = down_scaling
        self.scale_width_factor = scale_width_factor
        self.scale_height_factor = scale_height_factor
        self.scale_width_offset = scale_width_offset
        self.scale_height_offset = scale_height_offset
        
    def get_frame_shape(self):
        if self.down_scaling:
            return (self.mon.get("width") - self.scale_width_offset, 
                    self.mon.get("height") - self.scale_height_offset)
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
        if self.down_scaling:
            img = cv2.resize(img,dsize=(self.get_frame_shape()[0],self.get_frame_shape()[1]), fx=self.scale_width_factor,
                             fy=self.scale_height_factor, interpolation=cv2.INTER_AREA)
        
        return img
    
    def get_raw_frame(self, top_offset=41, left_offset=191, width_offset=382,
                 height_offset=518):
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
        #img = self.get_frame()
        #dead = 0
        while iterations > 0:
            # new_img = self.get_frame()
            new_img = self.get_raw_frame(41,191,382,518)
            cv2.imshow(title, new_img)
            # print(getProgress(new_img,0,274,0))
            iterations -= 1
            #if is_dead(img, new_img):
               # dead += 1
              #  print("Death: ", dead)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
            #img = new_img
    
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
        