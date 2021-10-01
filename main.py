from layer_predictions import Predictions
from layer_max_pooling import MaxPoolingLayer
from layer_softmax import SoftMax
from layer_relu import Relu
from keras.datasets import mnist
from layer_convolution import ConvolutionalLayer
from model_cnn import Model
import numpy as np
import cnn_helpers as helpers
import activation_functions as act
import utils
import error_functions as ef
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt
"""
During the backward phase, each layer will receive a gradient and also return a gradient. It will receive the gradient of loss with respect to its outputs (∂L / ∂out) and return the gradient of loss with respect to its inputs (∂L / ∂in).
"""
#28x28 pixel imgs
bias = 0
#Test is 10k long, train is 60k long.
(train_X, train_y), (test_X, test_y) = mnist.load_data()
#normalize:
train_X = (train_X / 255) - 0.5
test_X = (test_X / 255) - 0.5
train_y_one_hot_encoded = utils.one_hot_encode(train_y)
test_y_one_hot_encoded = utils.one_hot_encode(test_y)

#init layers:
learning_rate = 0.005 
num_filters = 1
filter_size = 5
train_img_num = 100
test_img_num = 50
epochs = 1
relu_leak_size = 0.01
i_between_valid = 1000

#resizing datasets for training/test/early stopping validation:
valid_X = train_X[train_img_num:train_img_num + 200] #We hardcore that we will classify 100 samples each time we validate with valid_X in regards to early stopping.
valid_y_one_hot_encoded = train_y_one_hot_encoded[train_img_num: train_img_num + 200]
train_X = train_X[:train_img_num]
train_y_one_hot_encoded = train_y_one_hot_encoded[:train_img_num]

test_X = test_X[:test_img_num]
test_y_one_hot_encoded = test_y_one_hot_encoded[:test_img_num]

#creating layers, init model:
conv = ConvolutionalLayer(learning_rate)
conv.init_filter(5, num_filters)
relu_conv = Relu(relu_leak_size)
max_pool = MaxPoolingLayer()
relu_pooling = Relu(relu_leak_size)
softmax = SoftMax(196 * num_filters, 10, learning_rate) #earlier: SoftMax(relu_pooling.output.size, 10)
predicts = Predictions()

kwargs = OrderedDict()
kwargs["conv"] = conv
#kwargs["relu_conv"] = relu_conv
kwargs["max_pool"] = max_pool
#kwargs["relu_pooling"] = relu_pooling
kwargs["softmax"] = softmax
kwargs["prediction"] = predicts
#init model, fit/test:
model_cnn = Model(kwargs)
iterations, accuracies, costs = model_cnn.fit(train_X, train_y_one_hot_encoded, valid_X, valid_y_one_hot_encoded, i_between_valid, epochs)
accuracy = model_cnn.test(test_X, test_y_one_hot_encoded)
#logging:
utils.log(accuracy, filter_size, num_filters, train_img_num, test_img_num, epochs, learning_rate, i_between_valid)

plt.plot(iterations, accuracies)
plt.xlabel("time")
plt.ylabel("accuracy")
plt.show()

#model_cnn.test(test_X[0:1000], test_y[0:1000])




