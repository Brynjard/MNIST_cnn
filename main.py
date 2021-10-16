from layer_predictions import Predictions
from layer_max_pooling import MaxPoolingLayer
from layer_softmax import SoftMax
from layer_relu import Relu
from keras.datasets import mnist
from layer_convolution import ConvolutionalLayer
from model_cnn import Model
import utils
from collections import OrderedDict
import matplotlib.pyplot as plt

#28x28 pixel images
bias = 0
#Test is 10k long, train is 60k long.
(train_X, train_y), (test_X, test_y) = mnist.load_data()
#normalize:
train_X = (train_X / 255) - 0.5
test_X = (test_X / 255) - 0.5
#one-hot encoding:
train_y_one_hot_encoded = utils.one_hot_encode(train_y)
test_y_one_hot_encoded = utils.one_hot_encode(test_y)

#init hyperparams:
learning_rate = 0.005 
num_filters = 1
filter_size = 5
train_img_num = 100
test_img_num = 50
epochs = 1
relu_leak_size = 0.01
i_between_valid = 1000

train_X = train_X[:train_img_num]
train_y_one_hot_encoded = train_y_one_hot_encoded[:train_img_num]

test_X = test_X[:test_img_num]
test_y_one_hot_encoded = test_y_one_hot_encoded[:test_img_num]

#creating layers, init model:
conv = ConvolutionalLayer(learning_rate)
conv.init_filter(5, num_filters)
#relu_conv = Relu(relu_leak_size)
max_pool = MaxPoolingLayer()
#relu_pooling = Relu(relu_leak_size)
softmax = SoftMax(196 * num_filters, 10, learning_rate)
predicts = Predictions()

kwargs = OrderedDict()
kwargs["conv"] = conv
kwargs["max_pool"] = max_pool
kwargs["softmax"] = softmax
kwargs["prediction"] = predicts

#init model, fit/test:
model_cnn = Model(kwargs)
iterations, accuracies, costs = model_cnn.fit(train_X, train_y_one_hot_encoded, epochs)
accuracy = model_cnn.test(test_X, test_y_one_hot_encoded)
#logging:
utils.log(accuracy, filter_size, num_filters, train_img_num, test_img_num, epochs, learning_rate, i_between_valid)
#Simple plot for visualization:
plt.plot(iterations, accuracies)
plt.xlabel("time")
plt.ylabel("accuracy")
plt.show()




