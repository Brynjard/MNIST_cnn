import numpy as np
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
            #Shuffle training data: 
            permutation = np.random.permutation(len(train_X))
            train_X = train_X[permutation]
            train_y = train_y[permutation]
            accuracy_counter = 0
            iterations = []
            accuracies = []
            costs = []

            for i, (im, label) in enumerate(zip(train_X, train_y)):
                accuracy, error = self.forward(im, label)
                accuracy_counter += accuracy
                if i % 10 == 0:
                    if accuracy_counter > 0 and i > 0:
                        current_accuracy = accuracy_counter / i

                        iterations.append(i)
                        accuracies.append(current_accuracy)
                        costs.append(error)
                    else:
                        current_accuracy = accuracy_counter
                    print("{} iterations gone. Accuracy: {}".format(i, current_accuracy))
                    print("Error: {}".format(error))
                self.backward()
        return iterations, accuracies, costs
    
    def test(self, test_X, test_y):
        print("TESTING MODEL")
        total_accuracy = 0
        total_loss = 0
        i = 0
        for im, label in zip(test_X, test_y):
            accuracy, error = self.forward(im, label)
            total_accuracy += accuracy
            total_loss += error
            i += 1
        print("*******************")
        print("Accuracy of model: {}".format(total_accuracy / i))
        print("*******************")



                






        
