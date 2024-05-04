from skimage.metrics import structural_similarity
import time


def is_dead(img, button_loc):
    #Uses the green color of the retry buttons to find the death state
    x, height = button_loc
    pixel = img[height][x]
    return pixel[1] > 160 and pixel[0] < pixel[1] and pixel[2] < pixel[1]

def is_dead_progress(img, range_start=0, range_end=274, height=0):
    if getProgress(img, range_start, range_end, height) == 0.0:
        return True
    return False
    
def getProgress(img, range_start, range_end, height):
    samples = range(range_start, range_end)
    height = height
    green_count = 0

    for x in samples:
        pixel = img[height][x]
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
    
