# import tensorflow as tf
# import tensorflow_datasets as tfds
import time
import network
# import network2
import mnist_loader

init_time = time.time()

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

start_time = time.time()

net = network.Network([784, 30, 10])

net.SGD(training_data, 30, 200, 0.1, test_data)

print("--- init_time %s seconds ---" % (time.time() - init_time))
print("--- start_time %s seconds ---" % (time.time() - start_time))

# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
#
# net.large_weight_initializer()
#
# net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
