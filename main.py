from layer_fully_connected import FullyConnectedLayer
from layer_output import OutputLayer
from keras.datasets import mnist
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
#Create filter:
filter = helpers.init_filter(5)
#convolve: 
convolved = helpers.convolve(train_X[0], filter)
#add bias: 
convolved = helpers.add_bias(convolved, bias)
#non-linearity:
non_lin = act.relu(convolved)
#pooling:
pooled_img = helpers.max_pooling(non_lin)
pooled_img = act.relu(pooled_img)
#Seems ok this far..
pooled_flattened = np.ndarray.flatten(pooled_img)

fc_layer = FullyConnectedLayer(pooled_flattened, helpers.init_weights(len(pooled_flattened), 10), np.zeros(10))
fc_layer.forward()
fc_layer.apply_activation(act.softmax)

print("Preds after softmax: {}".format(fc_layer.y))
output_layer = OutputLayer(fc_layer.y, train_y_one_hot_encoded[0])
output_layer.calc_error(ef.cross_entropy)
print("ERROR: {}".format(output_layer.error))




