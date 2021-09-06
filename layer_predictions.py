import numpy as np
import cnn_helpers as helpers
class Predictions():
    def __init__(self, pred, out_s):
        self.error = None
        self.pred = pred
        self.out_s = out_s
        self.d_L_d_out_s = None
        self.label = helpers.reverse_one_hot_encoding(self.out_s)

    def calc_error(self, error_func):
        l = - np.log(self.pred[self.label])
        self.error = l
        return self.error

    def backwards(self):
        #Calculate the dL / dout_s (softmax output) - this is the input for the softmax backwards function
        gradient = np.zeros(10)
        gradient[self.label] = -1 / self.pred[self.label]
        self.d_L_d_out_s = gradient
        self.print_desc()
        return self.d_L_d_out_s
    
    def print_desc(self):
        print("***** PREDICTION LAYER START ******")
        print("Softmax out: {}".format(self.out_s))
        print("Error: {}".format(self.error))
        print("d_L/d_out_s: {}".format(self.d_L_d_out_s))
        print("***** PREDICTION LAYER END ******")
    
    

    
    

    
    
        
