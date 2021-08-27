from keras.datasets import mnist
import numpy as np
import cnn_helpers as helpers
import activation_functions as act
from PIL import Image
#28x28 pixel imgs
"""if os.path.exists("/Users/brynjard/Documents/MNIST_cnn/data/train_X.npy"):
    train_X = np.load("/Users/brynjard/Documents/MNIST_cnn/data/train_X.npy")
    train_y = np.load("/Users/brynjard/Documents/MNIST_cnn/data/train_y.npy")
    test_X = np.load("/Users/brynjard/Documents/MNIST_cnn/data/test_X.npy")
    test_y = np.load("/Users/brynjard/Documents/MNIST_cnn/data/test_y.npy")
else:
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    np.save("/Users/brynjard/Documents/MNIST_cnn/data/train_X.npy", train_X)
    np.save("//Users/brynjard/Documents/MNIST_cnn/data/train_y.npy", train_y)
    np.save("//Users/brynjard/Documents/MNIST_cnn/data/test_X.npy", test_X)
    np.save("//Users/brynjard/Documents/MNIST_cnn/data/test_y.npy", test_y)"""

(train_X, train_y), (test_X, test_y) = mnist.load_data()
#Create filter:
filter = helpers.init_filter(5)
#convolve: 
convolved = helpers.convolve(train_X[0], filter)
#non-linearity:
non_lin = act.relu(convolved)
print("Image size after conv + non_linarity: {}".format(non_lin.shape))
print(non_lin)
#pooling:
pooled_img = helpers.max_pooling(non_lin)
