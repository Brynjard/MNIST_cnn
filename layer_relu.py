import numpy as np

class Relu():
    def __init__(self, x):
        self.x = x
        self.y = None
        self.gradient_o = None
        self.gradient_i = None
    
    def forward(self):
        self.y = np.where(self.x > 0, self.x, 0)
        return self.y

    def backwards(self):
        raise NotImplementedError
