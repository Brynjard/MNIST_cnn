class Layer():
    def __init__(self):
        self.input = None
        self.output = None
        self.weights = None
        self.bias = None

    def forward(self, input):
        raise NotImplementedError
    
    def backwards(self, error, eta):
        raise NotImplementedError
    
