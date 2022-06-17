
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# # Load data from https://www.openml.org/d/554
# X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
#
# # Divide by floating point to preserve decimals
# X = X / 255.
#
# # Rescale the data, use the traditional train/test split
# # Set train split to first 60000 elements and test split to everything after 60000
# X_train, X_test = X[:60000], X[60000:]
# y_train, y_test = y[:60000], y[60000:]

# Network initialization
# - Inputs (# of layers, # of nodes per layer, activation method, learning rate, backpropagation method, max # of epochs, loss function)
# - Initialize and store weights (matrix with input neuron # columns and output neuron # rows)
# - Initialize and store biases (vector with output neuron # rows)
# - Store activation methods
# - Store number of layers
# - Initialize number of neurons per layer
# - Store learning rate
class MyNN:
    def __init__(self, activation, learning_rate, node_layers):
        # These equations borrowed from http://neuralnetworksanddeeplearning.com/chap1.html
        # Aggregate pairs of values from node_layers and create arrays of random weights in those shapes
        # zip is much more concise than a standard for loop
        self.weights = [np.random.randn(y, x) for x, y in zip(node_layers[:-1], node_layers[1:])]
        # Create vectors of biases for each layer after the input layer
        self.biases = [np.random.randn(y, 1) for y in node_layers[1:]]

        self.activation = activation
        self.learning_rate = learning_rate
        self.node_layers = node_layers

    # This function borrowed from http://neuralnetworksanddeeplearning.com/chap1.html
    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))

    # This function borrowed from http://neuralnetworksanddeeplearning.com/chap1.html
    def feedforward(self, layer_input):
        # Loop through weights and biases, take dot product of input and weights,
        # add bias, squash the data and pass it to the next iteration
        for bias, weight in zip(self.biases, self.weights):
            # Take dot product of raw input and weight matrix and add the bias for the layer
            layer_input_adjusted = np.dot(weight, layer_input) + bias
            # Run the adjusted input through the sigmoid activation function to squash it
            layer_input = sigmoid(layer_input_adjusted)
        # Return final output
        return layer_input


new_nn = MyNN(node_layers=[1,5,3], activation='abc', learning_rate=0.1)

# Data training
# Data testing
# Backpropagation
# - Stochastic gradient descent
# - Batch sizes
