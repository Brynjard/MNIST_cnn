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
        total_preds = 0
        correct_preds = 0
        iterations = []
        accuracies = []
        costs = []
        for epoch in range(num_epochs):
            #Shuffle training data: 
            permutation = np.random.permutation(len(train_X))
            train_X = train_X[permutation]
            train_y = train_y[permutation]

            for im, label in zip(train_X, train_y):
                accuracy, error = self.forward(im, label)
                correct_preds += accuracy
                #performance metrics:
                if total_preds > 0:
                    accuracies.append(correct_preds / total_preds)
                else:
                    accuracies.append(correct_preds)
                iterations.append(total_preds)
                costs.append(error)

                if total_preds % 10 == 0 and total_preds > 0:
                    print("{} iterations done. Accuracy: {}".format(total_preds, correct_preds / total_preds))
                    print("Correct preds: {}".format(correct_preds))
                    print("Total preds: {}".format(total_preds))
                    print("Cost: {}".format(error))
                
                self.backward()
                total_preds += 1
        return iterations, accuracies, costs
    
    def test(self, test_X, test_y, log_string):
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
        #writing to logfile:
        f = open("log.txt", "a")
        f.write(log_string)
        f.write("\n")
        f.write("Accuracy of model: {}".format(total_accuracy / i))
        f.write("\n")
        f.close()



                






        
