import numpy as np
import cnn_helpers as helpers

class ConvolutionalLayer():
    def __init__(self, learning_rate):
        self.input = None
        self.output = None
        self.filter = None
        self.learning_rate = learning_rate

    def init_filter(self, filter_size):
        self.filter = helpers.init_filter(filter_size)
        return self.filter

    def forward(self, input):
        self.input = input
        self.output = helpers.convolve(self.input, self.filter)
        return self.output

    def backward(self, d_L_d_out):
        d_L_d_filters = np.zeros(self.filter.shape)
        d_L_d_input = np.zeros(self.input.shape)
        f_size = self.filter.shape[0]
        for r in range(self.input.shape[0] - 4):
            for c in range(self.input.shape[1]- 4):
                region = self.input[r:(f_size + r), c: (c + f_size)]
                d_L_d_filters += d_L_d_out[r, c] * region
                d_L_d_input[r:(f_size + r), c: (c + f_size)] += self.filter * d_L_d_out[r, c]
        self.filter -= self.learning_rate * d_L_d_filters
        self.d_L_d_filters = d_L_d_filters
        return self.d_L_d_filters



        

