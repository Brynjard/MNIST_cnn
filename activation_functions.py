
import numpy as np

def relu(x):
    return np.where(x > 0, x, 0)

def softmax(x):
    """print("*** SOFTMAX ***")
    print("X: {}".format(x))
    n = x - np.max(x)
    print("*** SOFTMAX END ***")
    return (np.exp(n) / sum(np.exp(n)))"""
    print("X: {}".format(x))
    n = np.exp(x - np.max(x))
    return n / np.sum(n)
