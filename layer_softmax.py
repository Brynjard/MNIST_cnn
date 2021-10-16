import numpy as np

class SoftMax():
    #input_len = number of nodes in input, nodes = number of nodes in output
    #This works as a fully connected layer with softmax. 
    def __init__(self, input_len, nodes, learning_rate):
        #We divide input by input_len to reduce the variance of our initial values
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.weights = self.weights.astype(np.float64)
        self.b = np.zeros(nodes, dtype=np.float64)
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
        totals = totals.astype(np.float64)
        self.last_totals = totals
        exp = np.exp(totals - np.max(totals))
        self.output = exp / np.sum(exp, axis=0)
        self.output = self.output.astype(np.float64)
        return self.output

    def backward(self, d_L_d_out):
        #d_L_d_out : Loss gradient for this layers output.
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue
            t_exp = np.exp(self.last_totals)
            t_exp = t_exp.astype(np.float64)
            S = np.sum(t_exp)
            S = S.astype(np.float64)
            #gradients of out[i] against totals:
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t.astype(np.float64)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            #Gradients of totals against weights/biases/input:
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            #Gradients of loss against totals:
            d_L_d_t = gradient * d_out_d_t
            d_L_d_t = d_L_d_t.astype(np.float64)
            #Gradients of loss against weights/biases/input:
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_w = d_L_d_w.astype(np.float64) 
            d_L_d_b = (d_L_d_t * d_t_d_b).astype(np.float64)
            d_L_d_inputs = (d_t_d_inputs @ d_L_d_t).astype(np.float64)

            self.weights -= self.learning_rate * d_L_d_w
            self.b -= self.learning_rate * d_L_d_b
            return (d_L_d_inputs.reshape(self.last_input_shape)).astype(np.float64)



