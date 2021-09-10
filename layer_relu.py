import numpy as np

class Relu():
    def __init__(self, input):
        self.input = input
        self.output = None
        self.d_L_d_input = None
    def forward(self):
        self.output = np.where(self.input > 0, self.input, 0)
        return self.output

    def backwards(self, d_L_d_out):
        #Derivative of relu is: if x <= 0 0, else 1
        self.d_L_d_input = np.where(d_L_d_out > 0, 1, 0)
        return self.d_L_d_input
