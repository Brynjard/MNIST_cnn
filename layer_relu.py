import numpy as np

class Relu():
    def __init__(self, leak_size):
        self.input = None
        self.output = None
        self.d_L_d_input = None
        self.leak_size = leak_size
    def forward(self, input):
        self.input = input
        self.output = (np.where(self.input > 0, self.input, self.leak_size * self.input)).astype(np.float64)
        return self.output


    def backward(self, d_L_d_out):
        #Derivative of relu is: if x <= 0 0, else 1
        self.d_L_d_input = (np.where(d_L_d_out > 0, 1, d_L_d_out * self.leak_size)).astype(np.float64)
        return self.d_L_d_input
