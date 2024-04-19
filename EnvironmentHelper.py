import numpy as np
from skimage.metrics import structural_similarity

DEAD_PATH = ".venv\\images\\dead.npy"
DEAD = np.load(DEAD_PATH)

def is_dead(img):
    similarity = structural_similarity(img,DEAD, win_size=3)
    
    if similarity > 0.92:
        return True
    else:
        return False