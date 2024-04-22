from skimage.metrics import structural_similarity

def is_dead(prev_img, cur_img):
    similarity = structural_similarity(prev_img,cur_img, win_size=3)
    if similarity < 0.91:
        return True
    else:
        return False