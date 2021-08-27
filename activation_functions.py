
import numpy as np

def relu(x):
    return np.where(x > 0, x, 0)
