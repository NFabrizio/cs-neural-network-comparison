import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Generate random sample data
# This code borrowed from https://beckernick.github.io/neural-network-scratch/
np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[3, .75],[.75, 3]], num_observations)
x2 = np.random.multivariate_normal([1, 6], [[2, .75],[.75, 2]], num_observations)
x3 = np.random.multivariate_normal([2, 12], [[1, .75],[.75, 1]], num_observations)

simulated_features = np.vstack((x1, x2, x3)).astype(np.float32)
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
# print(train_dataset[:5])
# print(train_labels[:5])
# print(train_labels[3000])
# print(test_dataset[:5])
# print(test_labels[:5])
# print(test_labels[148])
# print(train_dataset.shape)

# Create classifier
# This code from https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=70, alpha=1e-4,
                    solver='sgd', verbose=10,
                    learning_rate_init=.1)

# Fit the model to the data
mlp.fit(train_dataset, train_labels)

# Return mean accuracy on given data
print("Training set score: %f" % mlp.score(train_dataset, train_labels))
print("Test set score: %f" % mlp.score(test_dataset, test_labels))
