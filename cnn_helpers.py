import numpy as np

#Formula for calculating output layers size: (W−F+2P)/S+1
#Formula for calculating same size padding for stride = 1: P=(F−1)/2

def add_bias(matrix, b):
    return matrix + b
    

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
    return int(((w - f + 2*p) / s) + 1)

def apply_same_padding(image, p):
    #Assuming a stride of 1, calculating number of padding layers:
    p = int(p)
    padded = np.pad(image, [(p, p), (p, p)], mode="constant")
    return padded

def init_filter(filter_size, num_filters):
    #init filter to be random numbers within normal distribution
    return np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)

"""def init_weights(dim_0, dim1):
    #w = np.random.normal(size=(dim_0, dim1))
    #return w
    np.random.seed(0)
    scale = 1/max(1., (2+2)/2.)
    limit = np.sqrt(3.0 * scale)
    weights = np.random.uniform(-limit, limit, size=(dim_0,dim1))
    return weights"""

def init_bias(dim0, dim1):
    return np.zeros((1, dim1))

def convolve(image, filter, stride=1, same_padding=True):
    #Computing number of padding layers:
    p = compute_padding_layers_same_padding(filter.shape[0])
    if same_padding:
    #pad image:
        img_padded = apply_same_padding(image, p)
        output_layers = np.zeros((image.shape[0], image.shape[1], filter.shape[0]))
        for f in range(filter.shape[0]): #for all filters:
            f_size = filter.shape[1]
            for r in range(output_layers.shape[0]):
                if not (r + f_size > img_padded.shape[1]):
                    for c in range(output_layers.shape[0]):
                        if not (c + f_size > img_padded.shape[0]):
                            output_layers[r, c, f] = np.sum(filter[f] * img_padded[r:(f_size + r), c:f_size + c])
        return output_layers

    
def max_pooling(feature_matrix, filter_size=2, stride =2):
    """
    Max-pooling with 2x2 matrix and a stride of 2.
    """
    nums_r = feature_matrix.shape[0]
    nums_c = feature_matrix.shape[1]
    out_dim = calculate_output_size(feature_matrix.shape[0], filter_size, 0, stride)
    output = np.zeros((out_dim, out_dim), dtype=float)
    output_r = 0
    output_c = 0
    for r in range(0, nums_r, stride):
        for c in range(0, nums_c, stride):
            output[output_r, output_c] = np.amax(feature_matrix[r:r + filter_size, c:c + filter_size])
            if output_c >= out_dim - 1:
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

def reverse_one_hot_encoding(one_hot_encoded):
    "Reverses one-hot-encoded label to integer."
    return np.where(one_hot_encoded == 1)[0][0]



