from skimage.metrics import structural_similarity

def is_dead(prev_img, cur_img):
    similarity = structural_similarity(prev_img,cur_img, win_size=3)
    if similarity < 0.81:
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