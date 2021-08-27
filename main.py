from keras.datasets import mnist
import numpy as np
import cnn_helpers as helpers
import activation_functions as act
from PIL import Image
#currently following this: https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
#or this: https://www.linkedin.com/pulse/forward-back-propagation-over-cnn-code-from-scratch-coy-ulloa/
#28x28 pixel imgs
bias = 1.0
(train_X, train_y), (test_X, test_y) = mnist.load_data()
img = train_X[0]
im = Image.fromarray(img)
im.show()
#Create filter:
filter = helpers.init_filter(5)
#convolve: 
convolved = helpers.convolve(train_X[0], filter)
print(train_y[0])
#add bias: 
convolved = helpers.add_bias(convolved, bias)
#non-linearity:
non_lin = act.relu(convolved)
print("Image size after conv + non_linarity: {}".format(non_lin.shape))
non_lin_img = Image.fromarray(non_lin)
non_lin_img.show()
#pooling:
pooled_img = helpers.max_pooling(non_lin)
y = Image.fromarray(pooled_img)
y.show()
