import numpy as np
from layer import Layer


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        """
        The activation function is the function that we use to get the output of the neuron.

        The activation prime is the function that we use to get the derivative of the activation
        function.

        :param activation: The activation function
        :param activation_prime: The derivative of the activation function. Used in backpropagation
        """
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        """
        Given an input, return the output of the activation function

        :param input: The input to the layer
        :return: The activation function of the input.
        """
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        """
        Multiply the output gradient by the derivative of the activation function

        :param output_gradient: The gradient of the output layer, which is the gradient of the loss
        function
        :param learning_rate: The learning rate of the neural network
        :return: The gradient of the loss with respect to the input of the layer.
        """
        return np.multiply(output_gradient, self.activation_prime(self.input))
