import numpy as np
import cnn_helpers as helpers
class Predictions():
    def __init__(self, pred, targets):
        self.error = None
        self.pred = pred
        self.targets = targets
        self.gradient_o = None

    def calc_error(self, error_func):
        self.error = error_func(self.pred, self.targets)

    def backwards(self):
        #Calculate the dL / dout_s (softmax output) - this is the input for the softmax backwards function
        label = helpers.reverse_one_hot_encoding(self.targets)
        gradient = np.zeros(10)
        self.gradient_o = gradient[label] = -1 / self.pred[label]
        return self.gradient_o
    

    
    

    
    
        
