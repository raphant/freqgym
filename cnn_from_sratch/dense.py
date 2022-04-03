import numpy as np
from layer import Layer


class Dense(Layer):
    def __init__(self, input_size, output_size):
        """
        Initialize the weights and bias of the linear layer

        :param input_size: The number of features in the input data
        :param output_size: The number of output nodes in the layer
        """
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        """
        Given an input, return the dot product of the input and the weights plus the bias

        :param input: the input to the layer. This is the output from the previous layer, or for the
        first layer, the input to the network
        :return: The output of the layer.
        """
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        """
        Given the output gradient and the learning rate,
        update the weights and the bias

        :param output_gradient: The gradient of the output of the network
        :param learning_rate: The learning rate is a hyperparameter that controls how much we are
        adjusting the weights of our network with respect the loss gradient
        :return: The input_gradient
        """
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
