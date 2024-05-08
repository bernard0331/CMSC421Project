def is_dead(img, button_loc):
    #Uses the green color of the retry buttons to find the death state
    x, height = button_loc
    pixel = img[height][x]
    return pixel[1] > 160 and pixel[0] < pixel[1] and pixel[2] < pixel[1]


def getProgress(img, range_start, range_end, height):
    samples = range(range_start, range_end)
    height = height
    green_count = 0

    for x in samples:
        pixel = img[height][x]
        if(pixel[1] > 160 and pixel[0] < pixel[1] and pixel[2] < pixel[1]):
            green_count += 1
    
    return green_count/len(samples)

def is_dead_progress(progress, previous_progress):
    # Only consider dead if progress is zero and it was previously above 0%
    if progress == 0.0 and previous_progress > 0.0:
        return True
    return False