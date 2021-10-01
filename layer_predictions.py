import numpy as np
from numpy.core.numeric import NaN
import cnn_helpers as helpers
class Predictions():
    def __init__(self): #out_s : target for a single image.
        self.error = None
        self.d_L_d_out = None #formerly: d_L_d_out_s

    def forward(self, pred, out_s):
        self.pred = pred.astype(np.float64)
        self.out_s = out_s.astype(np.float64)
        self.label = helpers.reverse_one_hot_encoding(self.out_s)
        self.calc_error()
        accuracy = 1 if np.argmax(self.pred) == self.label else 0
        return accuracy, self.error

    def calc_error(self):
        l = - np.log(self.pred[self.label])
        self.error = l
        if np.isnan(self.error):
            print("NaN found in error: Predicted value for correct label ({}): {}".format(self.label, self.pred[self.label]))
        return self.error

    def backward(self):
        #Calculate the dL / dout_s (softmax output) - this is the input for the softmax backwards function
        gradient = (np.zeros(10)).astype(np.float64)
        gradient[self.label] = -1 / self.pred[self.label]
        self.d_L_d_out = gradient
        #self.print_desc()
        return self.d_L_d_out
    
    def print_desc(self):
        print("***** PREDICTION LAYER START ******")
        print("Output of model: {}".format(self.pred))
        print("target output: {}".format(self.out_s))
        print("Argmax of output: {}".format(np.argmax(self.pred)))
        print("Error: {}".format(self.error))
        #print("d_L/d_out_s: {}".format(self.d_L_d_out_s))
        #print("***** PREDICTION LAYER END ******")
    
    

    
    

    
    
        
