from layer_fully_connected import FullyConnectedLayer
from layer_output import OutputLayer
from layer_max_pooling import PoolingLayer
from keras.datasets import mnist
from layer_convolution import ConvolutionalLayer
import numpy as np
import cnn_helpers as helpers
import activation_functions as act
import utils
import error_functions as ef
from PIL import Image
#28x28 pixel imgs
bias = 0
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_y_one_hot_encoded = utils.one_hot_encode(train_y)
img = train_X[0]

#Convolutional layer:
conv_layer = ConvolutionalLayer(train_X[0], act.relu)
conv_layer.init_filter(5)
conv_layer.convolve()
convoluted_img = conv_layer.apply_activation_function()
#pooling:
pool_layer = PoolingLayer(convoluted_img, act.relu)
pool_layer.pool_layer()
pool_layer.apply_activation()
pooled_flattened = np.ndarray.flatten(pool_layer.y)

fc_layer = FullyConnectedLayer(pooled_flattened)
fc_layer.init_bias(10)
fc_layer.init_weights(10)
fc_layer.forward()
fc_layer.apply_activation(act.softmax)

print("Preds after softmax: {}".format(fc_layer.y))
output_layer = OutputLayer(fc_layer.y, train_y_one_hot_encoded[0])
output_layer.calc_error(ef.cross_entropy)
print("ERROR: {}".format(output_layer.error))




