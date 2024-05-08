import time


def is_dead(img, button_loc):
    #Uses the green color of the retry buttons to find the death state
    x, height = button_loc
    pixel = img[height][x]
    return pixel[1] > 160 and pixel[0] < pixel[1] and pixel[2] < pixel[1]

def is_dead_progress(img, previous_progress, range_start=0, range_end=106, height=0):
    progress = getProgress(img, range_start, range_end, height)
    # Only consider dead if progress is zero and it was previously above 0%
    if progress == 0.0 and previous_progress > 0.0:
        return True, progress
    return False, progress
    
def getProgress(img, range_start, range_end, height):
    total_pixels = range_end - range_start
    height = height
    green_count = 0

    for x in range(total_pixels):
        pixel = img[height][x]
        # Green is dominant and above a brightness threshold
        if(pixel[1] > 140 and pixel[0] < pixel[1] and pixel[2] < pixel[1]):
            green_count += 1
    
    # Calculate the percentage of green pixels
    progress_percentage = (green_count/total_pixels) * 100
    return round(progress_percentage,4)
    
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
    
