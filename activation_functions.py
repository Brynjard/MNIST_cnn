
import numpy as np

def relu(x):
    return np.where(x > 0, x, 0)

def softmax(x):
    #Returns a matrix of derivatives. 
    n = np.exp(x - np.max(x))
    return n / np.sum(n)

def softmax_derivative(y):
    #Will return a matrix of derivatives.
    return y * (1 - y)
