import numpy as np
import cnn_helpers as helpers
import activation_functions as af
from layer_interface import Layer

class PoolingLayer(Layer):
    def __init__(self, x, activation_function):
        self.x = x
        self.a = None
        self.y = None
        self.activation_function = activation_function
    
    def pool_layer(self):
        self.a = helpers.max_pooling(self.x)
        return self.a
    
    def apply_activation(self):
        self.y = af.relu(self.a)
        return self.y