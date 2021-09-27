import numpy as np

class SoftMax():
    #input_len = number of nodes in input, nodes = number of nodes in output
    #This works as a fully connected layer with softmax. 
    def __init__(self, input_len, nodes, learning_rate):
        #We divide input by input_len to reduce the variance of our initial values /(normalizing?)
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.b = np.zeros(nodes)
        self.output = None
        self.gradient_o = None
        self.learning_rate = learning_rate
    def forward(self, input):
        #Fully connected step:
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        input_len, shape = self.weights.shape
        totals = np.dot(input, self.weights) + self.b
        self.last_totals = totals
        ###

        """n = np.exp(totals - np.max(totals))
        self.output = n / np.sum(n)"""
        exp = np.exp(totals)
        self.output = exp / np.sum(exp, axis=0)
        return self.output

    def backward(self, d_L_d_out):
        #d_L_d_out : Loss gradient for this layers output.
        #We know only one element of d_L_d_out is nonzero:
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue
            t_exp = np.exp(self.last_totals)
            S = np.sum(t_exp)
            #gradients of out[i] against totals:
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            #Gradients of totals against weights/biases/input:
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            #Gradients of loss against totals:
            d_L_d_t = gradient * d_out_d_t

            #Gradients of loss against weights/biases/input:
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            #self.print_desc(d_L_d_out, d_out_d_t, self.last_input, d_t_d_b, d_t_d_inputs, d_t_d_w, d_L_d_t, d_L_d_inputs)
            self.weights -= self.learning_rate * d_L_d_w
            self.b -= self.learning_rate * d_L_d_b
            return d_L_d_inputs.reshape(self.last_input_shape)


    def print_desc(self, d_L_d_out, d_out_d_t, last_input, d_t_d_b, d_t_d_inputs, d_t_d_w, d_L_d_t, d_L_d_inputs):
        print("***** SOFTMAX LAYER START ******")
        print("d_L/d_out (loss gradient for this layers output): {}".format(d_L_d_out))
        print("d_out/d_t: {}".format(d_out_d_t))
        print("d_t/d_w: {}".format(last_input.shape))
        print("d_t/d_b: {}".format(d_t_d_b))
        print("d_t/d_inputs: {}".format(d_t_d_inputs.shape))
        print()
        print("d_L/d_w: {} @ {}".format(d_t_d_w[np.newaxis].T.shape, d_L_d_t[np.newaxis].shape))
        print("d_L/d_b: {} * {}".format(d_L_d_t.shape, d_t_d_b))
        print("d_L/d_inputs: {} @ {}".format(d_t_d_inputs.shape, d_L_d_t.shape))
        print("***** PREDICTION LAYER END ******")


