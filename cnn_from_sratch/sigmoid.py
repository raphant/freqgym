import numpy as np
from activation import Activation


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            """
            It takes in a value and returns a value between 0 and 1

            :param x: the input data
            :return: The sigmoid of x.
            """
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            """
            Return the derivative of the sigmoid function

            :param x: the input to the network
            :return: The derivative of the sigmoid function.
            """
            return sigmoid(x) * (1 - sigmoid(x))

        super().__init__(sigmoid, sigmoid_prime)
