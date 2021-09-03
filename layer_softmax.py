import numpy as np

class SoftMax():
    def __init__(self, x):
        self.x = x
        self.y = None

    def forward(self):
        n = np.exp(self.x - np.max(self.x))
        self.y = n / np.sum(n)
        return self.y

    def backwards(self):
        raise NotImplementedError
