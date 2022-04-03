import numpy as np
from layer import Layer
from scipy import signal


class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size: int, depth: int):
        """
        Given an input shape, a kernel size, and a depth, the function returns the output shape of the
        convolution layer

        :param input_shape: the depth, height, and width of the input
        :param kernel_size: The size of each matrix inside the kernel
        :param depth: The number of filters (kernels) and also the depth of the output
        """
        super().__init__()
        input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.depth = depth
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernel_shape = (depth, input_depth, kernel_size, kernel_size)

        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(*self.output_shape)

        print('input shape:', input_shape)
        print('kernel shape: ', self.kernels.shape)
        print('bias shape: ', self.biases.shape)
        print('output shape: ', self.output_shape)

    def forward(self, input):
        """
        Given an input, the function returns the output of the convolutional layer

        :param input: The input to the convolutional layer
        :return: The output of the convolutional layer
        """
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(
                    self.input[j], self.kernels[i][j], mode='valid'
                )
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Given an output gradient, compute the gradient of the weights and the gradient of the biases,
        and return the gradient of the input

        :param output_gradient: The gradient of the loss with respect to the output of the conv layer
        :param learning_rate: The learning rate is a hyperparameter that controls how much we are
        adjusting the weights of our network with respect the loss gradient. The lower the value, the
        slower we travel along the downward slope. While this might be a good idea (using a low learning
        rate) in terms of making sure that the network is learning, it is also a good idea to use a
        high learning rate, because the network will be learning faster.
        :return: The input_gradient
        """
        kernels_gradient = np.zeros(self.kernel_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(
                    self.input[j], output_gradient[i], mode='valid'
                )
                input_gradient[j] += signal.convolve2d(
                    output_gradient[i], self.kernels[i, j], mode='full'
                )
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient


if __name__ == '__main__':

    Convolutional((3, 3, 3), 3, 3)
