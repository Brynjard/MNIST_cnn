from layer_fully_connected import FullyConnectedLayer
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

img = train_X[0]
#init layers:

conv = ConvolutionalLayer(0.001)
conv.init_filter(5)

relu_conv = Relu()

max_pool = MaxPoolingLayer()

relu_pooling = Relu()

softmax = SoftMax(196, 10, 0.001) #earlier: SoftMax(relu_pooling.output.size, 10)

predicts = Predictions()
#Order layers for model:
kwargs = OrderedDict()
kwargs["conv"] = conv
kwargs["relu_conv"] = relu_conv
kwargs["max_pool"] = max_pool
kwargs["relu_pooling"] = relu_pooling
kwargs["softmax"] = softmax
kwargs["prediction"] = predicts

model_cnn = Model(kwargs)
#current best: 6450 iterations
iterations, accuracies, costs = model_cnn.fit(train_X[0:1000], train_y_one_hot_encoded[0:1000], 3)

model_cnn.test(test_X[:2000], test_y_one_hot_encoded[:2000])
plt.plot(iterations, accuracies)
plt.xlabel("time")
plt.ylabel("accuracy")
plt.show()
#model_cnn.forward(train_X[0], train_y_one_hot_encoded[0])
#model_cnn.backward()
#model_cnn.test(test_X[0:1000], test_y[0:1000])




