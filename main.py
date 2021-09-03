from layer_fully_connected import FullyConnectedLayer
from layer_predictions import Predictions
from layer_max_pooling import MaxPoolingLayer
from layer_softmax import SoftMax
from layer_relu import Relu
from keras.datasets import mnist
from layer_convolution import ConvolutionalLayer
import numpy as np
import cnn_helpers as helpers
import activation_functions as act
import utils
import error_functions as ef
from PIL import Image
"""
During the backward phase, each layer will receive a gradient and also return a gradient. It will receive the gradient of loss with respect to its outputs (∂L / ∂out) and return the gradient of loss with respect to its inputs (∂L / ∂in).
"""
#28x28 pixel imgs
bias = 0
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_y_one_hot_encoded = utils.one_hot_encode(train_y)
img = train_X[0]


#Convolutional layer:
conv = ConvolutionalLayer(train_X[0])
conv.init_filter(5)
conv.forwards()
relu = Relu(conv.y)
relu.forward()
#pooling:
max_pool = MaxPoolingLayer(relu.y)
max_pool.forward()
relu = Relu(max_pool.y)
relu.forward()

softmax = SoftMax(relu.y.size, 10)
softmax.forward(relu.y)
predicts = Predictions(softmax.y, train_y_one_hot_encoded[0])
predicts.calc_error(ef.cross_entropy)
predicts.backwards()



