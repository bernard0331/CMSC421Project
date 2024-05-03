from skimage.metrics import structural_similarity
import time

def is_dead(prev_img, cur_img, threshold=0.91):
    """Compared a screenshot of the previous frame to the screenshot of the current
    frame to determine if the agent has died or not.

    Args:
        prev_img (Matlike): A screenshot of the previous frame.
        cur_img (Matlike): A screenshot of the current frame.
        threshold (float): A value between 0 and 1.0 determing the sensitivity 
            of the method to differences in the two screenshots. A value closer
            to 1.0 makes it more sensitive to differences (could lead to false
            positives) and a value closer to 0 makes it less sensitive (could lead
            to false negatives). 

    Returns:
        bool: True if the agent is dead, False otherwise.
    """
    similarity = structural_similarity(prev_img,cur_img)
    if similarity < threshold:
        print("is_dead similarity: ",similarity)
        return True
    else:
        return False
    
def getProgress(img, range_start, range_end, height):
    samples = range(range_start, range_end)
    height = height
    green_count = 0

    for x in samples:
        pixel = img.pixel(x,height)
        if(pixel[1] > 160 and pixel[0] < pixel[1] and pixel[2] < pixel[1]):
            green_count += 1

    return green_count/len(samples)
    
def get_environment_fps(game):
    """Calculates the approximate fps of running the environment.

    Args:
        game (GDashEnv): An instance of a GDashEnv.

    Returns:
        int : The approximate frames per second.
    """
    game.reset()
    fps = 0
    initial_time = time.time()
    while time.time() - initial_time < 10:
        action = game.action_space.sample()
        game.step(action)
        fps += 1                         
    return fps/10
    
