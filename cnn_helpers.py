import numpy as np

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
    filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)
    filters = filters.astype(np.float64)
    return filters

def convolve(image, filter, stride=1, same_padding=True):
    p = compute_padding_layers_same_padding(filter.shape[0])
    if same_padding:
        img_padded = apply_same_padding(image, p)
        output_layers = np.zeros((image.shape[0], image.shape[1], filter.shape[0]))
        output_layers.astype(np.float64)
        for f in range(filter.shape[0]):
            f_size = filter.shape[1]
            for r in range(output_layers.shape[0]):
                if not (r + f_size > img_padded.shape[1]):
                    for c in range(output_layers.shape[0]):
                        if not (c + f_size > img_padded.shape[0]):
                            output_layers[r, c, f] = np.sum(filter[f] * img_padded[r:(f_size + r), c:f_size + c])
        return output_layers

def reverse_one_hot_encoding(one_hot_encoded):
    return np.where(one_hot_encoded == 1)[0][0]



