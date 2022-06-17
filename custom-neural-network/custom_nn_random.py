import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

class MyNN:
    def __init__(self, activation, learning_rate, node_layers):
        # self.weights
        # self.biases
        self.activation = activation
        self.learning_rate = learning_rate
        self.node_layers = node_layers

    def activation_function(self):
        if self.activation == 'sgd':
            print('sgd activation')
        elif self.activation == 'relu':
            print('relu activation')
        else:
            print('default activation')

    def train(self, data):
        # Create vector of 0-value biases and random weights between 0 and 1
        bias_array = []
        weights_array = []

        for i in range(len(node_layers) - 1):
            # Bias vector should be 1 x number_of_nodes_in_next_layer
            bias_array.append(np.zeros(node_layers[i + 1]))

            # Weights matrix should be number_of_nodes_in_this_layer x number_of_nodes_in_next_layer
            weights_array.append(np.random.uniform(size=(node_layers[i], node_layers[i + 1])))

        self.biases = np.asarray(bias_array, dtype=object)
        self.weights = np.asarray(weights_array, dtype=object)

        # print(self.biases)
        # print(self.weights)

        print('self.weights[0]')
        print(self.weights[0])

        first_output = np.dot(data, self.weights[0]) + self.biases[0]
        print('first_output')
        print(first_output)

        self.activation_function()

new_nn = MyNN(node_layers=[1,5,3], activation='abc', learning_rate=0.1)
new_nn.train(np.array([[1], [2], [3], [4], [5]]))

# Generate random sample data
# This code borrowed from https://beckernick.github.io/neural-network-scratch/
np.random.seed(12)
num_observations = 5000

# Create 3 random data sets
x1 = np.random.multivariate_normal([0, 0], [[3, .75],[.75, 3]], num_observations)
x2 = np.random.multivariate_normal([1, 6], [[2, .75],[.75, 2]], num_observations)
x3 = np.random.multivariate_normal([2, 12], [[1, .75],[.75, 1]], num_observations)

# Stack the 3 data sets vertically
simulated_features = np.vstack((x1, x2, x3)).astype(np.float32)
# Stack the labels horizontally
simulated_labels = np.hstack((np.zeros(num_observations),
				np.ones(num_observations), np.ones(num_observations) + 1))

plt.figure(figsize=(12,8))
plt.scatter(simulated_features[:, 0], simulated_features[:, 1],
            c = simulated_labels, alpha = .4)
# plt.show()

# Prepare data
# This code borrowed from https://beckernick.github.io/neural-network-scratch/
labels_onehot = np.zeros((simulated_labels.shape[0], 3)).astype(int)
labels_onehot[np.arange(len(simulated_labels)), simulated_labels.astype(int)] = 1

train_dataset, test_dataset, train_labels, test_labels = train_test_split(
    simulated_features, labels_onehot, test_size = .1, random_state = 12)
# train_dataset, test_dataset = simulated_features[:3500], simulated_features[3500:]
# train_labels, test_labels = simulated_labels[:3500], simulated_labels[3500:]
print(train_dataset[:5])
print(train_dataset.shape[1])
# print(train_labels[:5])
# print(train_labels[3000])
# print(test_dataset[:5])
# print(test_labels[:5])
# print(test_labels[148])
