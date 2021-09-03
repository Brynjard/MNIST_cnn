import numpy as np

class SoftMax():
    #input_len = number of nodes in input, nodes = number of nodes in output
    #This works as a fully connected layer with softmax. 
    def __init__(self, input_len, nodes):
        #We divide input by input_len to reduce the variance of our initial values /(normalizing?)
        self.weights = np.random.randn(input_len, nodes) / input_len 
        self.b = np.zeros(nodes)
        self.y = None
        self.gradient_o = None

    def forward(self, input):
        #Fully connected step:
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        input_len, shape = self.weights.shape
        totals = np.dot(input, self.weights) + self.b
        ###

        n = np.exp(totals - np.max(totals))
        self.y = n / np.sum(n)
        return self.y

    def backwards(self, gradient_i):
        #gradient_i : Loss gradient for this layers output.
        #We know only one element of gradient_i is nonzero:
        for i, gradient in enumerate(gradient_i):
            if gradient == 0:
                continue
            t_exp = np.exp(self.last_totals)

