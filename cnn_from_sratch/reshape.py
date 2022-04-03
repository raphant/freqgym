import numpy as np
from layer import Layer


class Reshape(Layer):
    def __init__(self, input_shape=None, output_shape=None):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x):
        """
        Given an input vector x, return a vector of the same shape as x

        :param x: The input to the layer
        :return: The input array reshaped to the shape of the output_shape.
        """
        return np.reshape(x, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        """
        Given the gradient of the loss with respect to the output of the layer,
        and given the learning rate, return the gradient of the loss with respect to the input of the
        layer

        :param output_gradient: The gradient of the loss function with respect to the output of the conv
        layer
        :param learning_rate: The learning rate of the neural network
        :return: The gradient of the loss with respect to the input of the layer.
        """
        return np.reshape(output_gradient, self.input_shape)
