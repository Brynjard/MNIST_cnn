import numpy as np
from layer_interface import Layer
import cnn_helpers as helpers

class ConvolutionalLayer(Layer):
    def __init__(self, x, activation_function):
        self.x = x
        self.a = None
        self.y = None
        self.filter = None
        self.activation_function = activation_function

    def init_filter(self, filter_size):
        self.filter = helpers.init_filter(filter_size)
        return self.filter

    def convolve(self):
        self.a = helpers.convolve(self.x, self.filter)
        return self.a

    def apply_activation_function(self):
        self.y = self.activation_function(self.a)
        return self.y

