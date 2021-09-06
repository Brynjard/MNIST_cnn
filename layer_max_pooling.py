import numpy as np
import cnn_helpers as helpers
import activation_functions as af
import numpy as np
class MaxPoolingLayer():
    def __init__(self, input, filter_size = 2, stride = 2):
        self.input = input
        self.y = None
        self.filter_size = filter_size
        self.stride = stride
    
    def forward(self):
        nums_r = self.input.shape[0]
        nums_c = self.input.shape[1]
        out_dim = helpers.calculate_output_size(self.input.shape[0], self.filter_size, 0, self.stride)
        output = np.zeros((out_dim, out_dim), dtype=float)
        output_r = 0
        output_c = 0
        for r in range(0, nums_r, self.stride):
            for c in range(0, nums_c, self.stride):
                output[output_r, output_c] = np.amax(self.input[r:r + self.filter_size, c:c + self.filter_size])
                if output_c >= out_dim - 1:
                    output_c = 0
                else:
                    output_c += 1
            output_r += 1
        self.y = output
        return self.y
    
    def backwards(self, d_L_d_out):
        #d_L_d_out is the loss gradient for this layers output
        d_L_d_input = np.zeros((self.input.shape))

        for r in range(self.input.shape[0]):
            for c in range(self.input.shape[1]):
                reg = self.input[r * 2: r * 2 + 2, c * 2: c * 2 + 2]
                reg_max = np.amax(reg)
                if self.input[r, c] == reg_max:
                    d_L_d_input[r, c] = reg_max
        return d_L_d_input

