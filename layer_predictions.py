import numpy as np
import cnn_helpers as helpers
class Predictions():
    def __init__(self, pred, targets):
        self.error = None
        self.pred = pred
        self.targets = targets
        self.gradient_o = None
        self.label = helpers.reverse_one_hot_encoding(self.targets)

    def calc_error(self, error_func):
        l = - np.log(self.pred[self.label])
        self.error = l
        return self.error

    def backwards(self):
        #Calculate the dL / dout_s (softmax output) - this is the input for the softmax backwards function
        gradient = np.zeros(10)
        gradient[self.label] = -1 / self.pred[self.label]
        self.gradient_o = gradient
        print("TARGETS: {}".format(self.targets))
        print("GRADIENT_O: {}".format(self.gradient_o))
        return self.gradient_o
    

    
    

    
    
        
