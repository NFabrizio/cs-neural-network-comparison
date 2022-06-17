# This code from https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html

import warnings
import time

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

print(__doc__)
init_time = time.time()

# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Divide by floating point to preserve decimals
X = X / 255.

# Rescale the data, use the traditional train/test split
# Set train split to first 60000 elements and test split to everything after 60000
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Set random to 1 for reproducible results, use 1e-4 for penalty
# mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
#                     solver='sgd', verbose=10, random_state=1,
#                     learning_rate_init=.1)
start_time = time.time()

# activation can be logistic, tanh or relu as well others
mlp = MLPClassifier(hidden_layer_sizes=(30,), max_iter=30, alpha=1e-4,
                    solver='sgd', activation='relu', verbose=10,
                    learning_rate_init=0.1, batch_size=200)

# This example won't converge because of CI's time constraints, so we catch the
# warning and ignore it here
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=ConvergenceWarning,
#                             module="sklearn")
#     mlp.fit(X_train, y_train)

mlp.fit(X_train, y_train)

# Return mena accuracy on given data
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))
print("--- init_time %s seconds ---" % (time.time() - init_time))
print("--- start_time %s seconds ---" % (time.time() - start_time))

# fig, axes = plt.subplots(4, 4)
# # use global min / max to ensure all weights are shown on the same scale
# vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
# for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
#     ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
#                vmax=.5 * vmax)
#     ax.set_xticks(())
#     ax.set_yticks(())
#
# plt.show()
