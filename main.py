from keras.datasets import mnist
import numpy as np
import cnn_helpers as helpers
import activation_functions as act
from PIL import Image
#28x28 pixel imgs

(train_X, train_y), (test_X, test_y) = mnist.load_data()
img = train_X[0]
im = Image.fromarray(img)
im.show()
#Create filter:
filter = helpers.init_filter(5)
#convolve: 
convolved = helpers.convolve(train_X[0], filter)
#non-linearity:
non_lin = act.relu(convolved)
print("Image size after conv + non_linarity: {}".format(non_lin.shape))
non_lin_img = Image.fromarray(non_lin)
non_lin_img.show()
#pooling:
pooled_img = helpers.max_pooling(non_lin)
y = Image.fromarray(pooled_img)
y.show()
