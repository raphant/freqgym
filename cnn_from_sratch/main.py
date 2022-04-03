from cnn import Convolutional
from dense import Dense
from keras.datasets import mnist
from reshape import Reshape
from sigmoid import Sigmoid

from cnn_from_sratch.layer import Layer
from cnn_from_sratch.preprocess import preprocess_data
from cnn_from_sratch.util import binary_cross_entropy, binary_cross_entropy_prime
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D

network: list[Layer] = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid(),
]

epochs = 20
learning_rate = 0.1
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# train
for e in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)
            # input(
            #     f'Just completed forward pass for {layer.__class__.__name__} layer | Shape of output: {output.shape}.\n'
            #     f'Head of output: {output[0,0]}\n'
            #     f' Press enter to continue.'
            # )
        # error
        error += binary_cross_entropy(y, output)

        # backward
        grad = binary_cross_entropy_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
            # input(
            #     f'Just completed backward pass for {layer.__class__.__name__} | Shape of grad: {grad.shape}. Press enter to continue.'
            # )

    error /= len(x_train)
    print(f"Epoch {e + 1}/{epochs}, error={error}")

# test
guesses: list[tuple] = []
for x, y in zip(x_test, y_test):
    # forward
    output = x
    for layer in network:
        output = layer.forward(output)
    guesses.append((output, y))
    print(f"Predicted: {output.argmax()}, Actual: {y.argmax()}")
number_correct = len([guess for guess in guesses if guess[0].argmax() == guess[1].argmax()])
print(
    f'Accuracy: {sum([1 for (pred, actual) in guesses if pred.argmax() == actual.argmax()]) / len(guesses)}, {number_correct}/{len(guesses)} correct'
)
