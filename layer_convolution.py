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
        print("d_L_d_out shape: {}".format(d_L_d_out.shape))
        for r in range(0, self.input.shape[0], self.filter.shape[0]):
            for c in range(0, self.input.shape[1], self.filter.shape[0]):
                



        

