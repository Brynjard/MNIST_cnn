class Model(object):
    def __init__(self, layers):
        self.layers = layers
        self.layer_keys = [k for k in self.layers.keys()]
    def forward(self, image, target):
        """
            Assuming that first layer is convolutional layer(takes image as input) and that last layer is Prediction-layer(takes output + target as input)
            image: numpy 2d array of image we are training on.
            target: 10-size vector with one-hot-encoded target.
        """        
        output = self.layers[self.layer_keys[0]].forward(image) #convolutional layer
        for i in range(1, len(self.layer_keys) - 1):
            output = self.layers[self.layer_keys[i]].forward(output)
        #Prediction layer:
        accuracy, error = self.layers[self.layer_keys[-1]].forward(output, target)
        return accuracy, error
    def backward(self):
        gradient = self.layers[self.layer_keys[-1]].backward()
        for layer_key in reversed(self.layer_keys[:-1]):
            gradient = self.layers[layer_key].backward(gradient)

    def fit(self, train_X, train_y, num_epochs=1):
        print("TRAINING MODEL")
        for epoch in range(num_epochs):
            for i, (im, label) in enumerate(zip(train_X, train_y)):
                accuracy, error = self.forward(im, label)
                self.backward()
    
    def test(self, test_X, test_y):
        print("TESTING MODEL")
        total_accuracy = 0
        total_loss = 0
        for im, label in zip(test_X, test_y):
            accuracy, error = self.forward(im, label)
            total_accuracy += accuracy
            total_loss += error
        print("*******************")
        print("Accuracy of model: {}".format(total_accuracy / len(test_X)))
        print("Avg. Loss: {}".format(total_loss / len(test_X)))
        print("*******************")



                






        
