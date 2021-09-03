import numpy as np
import cnn_helpers as helpers
import activation_functions as af

class MaxPoolingLayer():
    def __init__(self, x):
        self.x = x
        self.y = None
    
    def forward(self, filter_size=2, stride =2):
        nums_r = self.x.shape[0]
        nums_c = self.x.shape[1]
        out_dim = helpers.calculate_output_size(self.x.shape[0], filter_size, 0, stride)
        output = np.zeros((out_dim, out_dim), dtype=float)
        output_r = 0
        output_c = 0
        for r in range(0, nums_r, stride):
            for c in range(0, nums_c, stride):
                output[output_r, output_c] = np.amax(self.x[r:r + filter_size, c:c + filter_size])
                if output_c >= out_dim - 1:
                    output_c = 0
                else:
                    output_c += 1
            output_r += 1
        self.y = output
        return self.y
    
    def backwards(self):
        raise NotImplementedError
