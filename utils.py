import numpy as np

def one_hot_encode(target_vector):
    target_matrix = np.zeros((len(target_vector), 10))
    for t in range(10):
        is_class = target_vector == t
        is_class = is_class.astype("int")

        target_matrix[:, t] = is_class
    return target_matrix
    