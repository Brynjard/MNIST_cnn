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

    def backwards(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filter.shape)
        d_L_d_input = np.zeros(self.x.shape)
        f_size = self.filter.shape[0]
        for r in range(self.x.shape[0] - 4):
            for c in range(self.x.shape[1]- 4):
                region = self.x[r:(f_size + r), c: (c + f_size)]
                d_L_d_filters += d_L_d_out[r, c] * region
                d_L_d_input[r:(f_size + r), c: (c + f_size)] += self.filter * d_L_d_out[r, c]
        self.filter -= learn_rate * d_L_d_filters
        self.d_L_d_filters = d_L_d_filters
        return self.d_L_d_filters



        

