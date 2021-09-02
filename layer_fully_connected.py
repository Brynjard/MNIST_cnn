from network_layer import Layer

import numpy as np
#Denotes input vector  = x, acctivations as = y, and output as y
class FullyConnectedLayer(Layer):
    def __init__(self, x, w, bias):
        self.x = x
        self.a = None
        self.w = w
        self.y = None
        self.bias = bias

    def forward(self):
        #Bias gets added to every neuron:
        self.a = np.dot(self.x, self.w) + self.bias
        return self.a

    def apply_activation(self, activation_function):
        self.y = activation_function(self.a)
        return self.y

    def backwards(self, output_error, eta):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.x.T, output_error)

        self.weights -= eta * weights_error
        self.bias -= eta * output_error
        return input_error
