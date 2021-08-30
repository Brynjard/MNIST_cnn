from layer_fully_connected import FullyConnectedLayer
from layer_output import OutputLayer
from keras.datasets import mnist
import numpy as np
import cnn_helpers as helpers
import activation_functions as act
import utils
from PIL import Image
#currently following this: https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
#or this: https://www.linkedin.com/pulse/forward-back-propagation-over-cnn-code-from-scratch-coy-ulloa/
#28x28 pixel imgs
bias = 1.0
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_y_one_hot_encoded = utils.one_hot_encode(train_y)
img = train_X[0]
im = Image.fromarray(img)
#im.show()
#Create filter:
filter = helpers.init_filter(5)
#convolve: 
convolved = helpers.convolve(train_X[0], filter)
print(train_y[0])
#add bias: 
convolved = helpers.add_bias(convolved, bias)
#non-linearity:
non_lin = act.relu(convolved)
non_lin_img = Image.fromarray(non_lin)
#non_lin_img.show()
#pooling:
pooled_img = helpers.max_pooling(non_lin)
y = Image.fromarray(pooled_img)
#y.show()
pooled_flattened = np.ndarray.flatten(pooled_img)
fc_layer = FullyConnectedLayer(pooled_flattened, helpers.init_weights(len(pooled_flattened), 10), np.ones(10))
fc_layer.forward()
print("Preds before activation: {}".format(fc_layer.a))
fc_layer.apply_activation(act.softmax)
print("fc_layer x: {}".format(pooled_flattened.shape))
print("Shape of fc_layer output: {}".format(fc_layer.y.shape))
print("Shape of y: {} shape of bias: {} shape of weights: {}".format(fc_layer.y.shape, fc_layer.bias.shape, fc_layer.w.shape))
print("Predictions: {}".format(fc_layer.y))
output_layer = OutputLayer(fc_layer.y, train_y_one_hot_encoded[0])



