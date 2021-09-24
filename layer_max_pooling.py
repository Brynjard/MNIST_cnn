import numpy as np
import cnn_helpers as helpers
import activation_functions as af
import numpy as np
class MaxPoolingLayer():
    def __init__(self, filter_size = 2, stride = 2):
        self.input = None
        self.output = None
        self.filter_size = filter_size
        self.stride = stride
        self.d_L_d_input = None
    
    def forward(self, input):
        self.input = input

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
        self.output = output
        return self.output
    
    def backward(self, d_L_d_out):
        #d_L_d_out is the loss gradient for this layers output
        d_L_d_input = np.zeros((self.input.shape))
        out_dim = d_L_d_out.shape[0]
        nums_r = self.input.shape[0]
        nums_c = self.input.shape[1]
        output_r = 0
        output_c = 0
        for r in range(0, nums_r, self.stride):
            for c in range(0, nums_c, self.stride):
                curr_region_max = np.amax(self.input[r:r + self.filter_size, c:c + self.filter_size])
                for inner_r in range(r, r + self.filter_size):
                    for inner_c in range(c, c + self.filter_size):
                        if self.input[inner_r, inner_c] == curr_region_max and curr_region_max > 0:
                            d_L_d_input[inner_r, inner_c] = d_L_d_out[output_r, output_c]
                if output_c >= out_dim - 1:
                    output_c = 0
                else:
                    output_c += 1
            output_r += 1 
        self.d_L_d_input = d_L_d_input
        #self.print_desc()
        return self.d_L_d_input
    
    def print_desc(self):
        print("***** MAXPOOLING LAYER START *****")
        print("Output: {}".format(self.input))
        print("d_L_d_input: {}".format(self.d_L_d_input))
        print("***** MAXPOOLING LAYER END *****")

        

