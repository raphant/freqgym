import numpy as np


def binary_cross_entropy(y_true, y_pred):
    """
    Given a true value and a predicted value, compute the binary cross entropy

    :param y_true: The actual truth values of the data
    :param y_pred: The prediction of the network
    :return: The mean of the binary cross entropy between y_true and y_pred
    """
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_prime(y_true, y_pred):
    """
    The derivative of the binary cross entropy function is the difference between the probability of the
    true class and the probability of the predicted class

    :param y_true: The actual truth values of the input data
    :param y_pred: The output of the network
    :return: The derivative of the binary cross entropy loss function.
    """
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
