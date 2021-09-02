import numpy as np
from network_layer import Layer

class OutputLayer():
    def __init__(self, pred, targets):
        self.error = None
        self.pred = pred
        self.targets = targets

    def calc_error(self, error_func):
        self.error = error_func(self.pred, self.targets)
    
    

    
    
        
