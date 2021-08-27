from network_layer import Layer

import numpy as np
class FullyConnectedLayer(Layer):
    def __init__(self, input, weights, bias):
        self.input = input
        self.output = None
        self.weights = weights
        self.bias = bias

    def forward(self):
        #Bias gets added to every neuron:
        self.output = np.dot(self.input, self.weights) + self.bias

    def backwards(self, error, eta):
        
