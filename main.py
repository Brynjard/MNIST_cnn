from keras.datasets import mnist
import numpy as np
import pickle as pkl
import os
#28x28 pixel imgs
if os.path.exists("./data/train_X.npy"):
    train_X = np.load("data/train_X.npy")
    train_y = np.load("data/train_y.npy")
    test_X = np.load("data/test_X.npy")
    test_y = np.load("data/test_y.npy")
else:
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    np.save("./data/train/train_X.npy", train_X)
    np.save("./data/train/train_y.npy", train_y)
    np.save("./data/train/test_X.npy", test_X)
    np.save("./data/train/test_y.npy", test_y)


print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

print("Y_train[0]: {}".format(train_y[0]))
