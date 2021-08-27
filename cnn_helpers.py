import numpy as np

#Formula for calculating output layers size: (W−F+2P)/S+1
#Formula for calculating same size padding for stride = 1: P=(F−1)/2

def compute_padding_layers_same_padding(filter_size):
    p = (filter_size - 1) / 2

    if np.floor(p) - p != 0:
        p = np.ceil(p)
    return p

def calculate_output_size(w, f, p, s):
    """
    w = input size
    f = filtersize
    p = padding layers 
    s = stride
    """
    return (w - f + 2*p) / (s + 1)

def apply_same_padding(image, p):
    #Assuming a stride of 1, calculating number of padding layers:
    p = int(p)
    padded = np.pad(image, [(p, p), (p, p)], mode="constant")
    return padded

def init_filter(filter_size):
    #init filter to be random numbers within normal distribution
    return np.random.normal(size=(filter_size, filter_size))

def convolve(image, filter, stride=1, same_padding=True):
    #Computing number of padding layers:
    p = compute_padding_layers_same_padding(filter.shape[0])
    if same_padding:
    #pad image:
        output_layer = np.zeros((image.shape))
        image = apply_same_padding(image, p)
        f_size = filter.shape[0]
        for r in range(output_layer.shape[0]):
            if not (r + f_size > image.shape[1]):
                for c in range(output_layer.shape[0]):
                    if not (c + f_size > image.shape[0]):
                        output_layer[r, c] = np.sum(filter * image[r:(f_size + r), c:f_size + c])
    return output_layer
    
def max_pooling(feature_matrix):
    """
    Max-pooling with 2x2 matrix and a stride of 2.
    """
    nums_r = feature_matrix.shape[0]
    nums_c = feature_matrix.shape[1]
    output = np.zeros((nums_r // 2, nums_c // 2), dtype=float)
    output_r = 0
    output_c = 0
    for r in range(0, nums_r, 2):
        for c in range(0, nums_c, 2):
            output[output_r, output_c] = np.amax(feature_matrix[r:r + 2, c:c + 2])
            if output_c >= 74:
                output_c = 0
            else:
                output_c += 1
        output_r += 1
    return output

def avg_pooling(feature_matrix):
    """
    Max-pooling with 2x2 matrix and a stride of 2.
    """
    nums_r = feature_matrix.shape[0]
    nums_c = feature_matrix.shape[1]
    output = np.zeros((nums_r // 2, nums_c // 2), dtype=float)
    output_r = 0
    output_c = 0
    for r in range(0, nums_r, 2):
        for c in range(0, nums_c, 2):
            output[output_r, output_c] = feature_matrix[r:r + 2, c:c + 2].np.mean()
            if output_c >= 74:
                output_c = 0
            else:
                output_c += 1
        output_r += 1
    return output



