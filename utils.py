import numpy as np

def one_hot_encode(target_vector):
    target_matrix = np.zeros((len(target_vector), 10))
    for t in range(10):
        is_class = target_vector == t
        is_class = is_class.astype("int")

        target_matrix[:, t] = is_class
    return target_matrix

def log(accuracy, f_size, num_filters, num_training_imgs, num_test_imgs, epochs, learning_rate, i_between_validation):
    log_string = "\n******** \ntraining on {} images, for {} epochs. Testing with: {} images. \nlearning_rate:{} filter_size:{} number of filters: {} i between valdiation: {}\nAccuracy of model: {}\n".format(num_training_imgs, epochs, num_test_imgs, learning_rate, f_size, num_filters, i_between_validation, accuracy)
    f = open("log.txt", "a")
    f.write(log_string)
    f.close()





    