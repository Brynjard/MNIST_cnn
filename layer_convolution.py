import numpy as np
import cnn_helpers as helpers

class ConvolutionalLayer():
    def __init__(self, x):
        self.x = x
        self.y = None
        self.filter = None

    def init_filter(self, filter_size):
        self.filter = helpers.init_filter(filter_size)
        return self.filter

    def forwards(self):
        self.y = helpers.convolve(self.x, self.filter)
        return self.y

    def backwards(self):
        raise NotImplementedError

