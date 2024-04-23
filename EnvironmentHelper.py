from skimage.metrics import structural_similarity

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
    similarity = structural_similarity(prev_img,cur_img, win_size=3)
    if similarity < threshold:
        return True
    else:
        return False