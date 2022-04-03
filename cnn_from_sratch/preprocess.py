import numpy as np
from keras.utils import np_utils


def preprocess_data(x, y, limit):
    """
    Preprocess the data by:
        1. Selecting the first limit number of images with label 0 and limit number of images with label
        2. Shuffle the data.
        3. Reshape the data.
        4. Normalize the data.
        5. Convert the labels to one-hot encoding.

    :param x: The input data
    :param y: the labels of the images
    :param limit: the number of images to use
    :return: the preprocessed data.
    """
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]

    # the shape of the image
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype(np.float32) / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)

    return x, y
