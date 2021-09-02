
import numpy as np

def relu(x):
    return np.where(x > 0, x, 0)

def softmax(x):
    n = np.exp(x - np.max(x))
    return n / np.sum(n)
