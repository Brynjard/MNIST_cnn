import cnn_helpers as helpers
import numpy as np
#Denotes input vector  = x, acctivations as = y, and output as y
class FullyConnectedLayer():
    def __init__(self, x):
        if x.ndim > 1:
            self.x = np.ndarray.flatten(x)
        else:
            self.x = x
        self.w = None
        self.y = None
        self.bias = None

    def init_bias(self, next_layer_size):
        self.bias = np.zeros(next_layer_size)

    def init_weights(self, next_layer_size):
        self.w = helpers.init_weights(len(self.x), next_layer_size)

    def forward(self):
        #Bias gets added to every neuron:
        self.y = np.dot(self.x, self.w) + self.bias
        return self.y

    def backwards(self, output_error, eta):
        raise NotImplementedError
