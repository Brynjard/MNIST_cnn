
import numpy as np

def relu(x):
    return np.where(x > 0, x, 0)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 0)