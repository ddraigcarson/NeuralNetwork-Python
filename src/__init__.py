import random
import math
import time
import numpy as np
from pandas import Series
import struct

# Definitions
INPUT_NEURONS = 28*28
HIDDEN_LAYER_1_NEURONS = 70
HIDDEN_LAYER_2_NEURONS = 35
OUTPUT_NEURONS = 10

NETWORK_LAYERS = [INPUT_NEURONS, HIDDEN_LAYER_1_NEURONS, HIDDEN_LAYER_2_NEURONS, OUTPUT_NEURONS]

NETWORK_TRAINING_SET_START = 0
NETWORK_TRAINING_SET_END = 99
NETWORK_TESTING_SET_START = 59900
NETWORK_TESTING_SET_END = 59999

IMAGE_SIZE_BYTES = 784


class Network:
    timer = None

    NO_OF_LAYERS = 0
    OUTPUT_LAYER = 0

    layers = []
    neuron_outputs = []
    neuron_biases = []
    neuron_weights = []
    neuron_error_signals = []
    data_builder = None

    def __init__(self, layers, data_builder, timer):
        self.timer = timer
        self.timer.network_init_start = self.timer.log_time()

        self.layers = layers
        self.NO_OF_LAYERS = len(layers)
        self.OUTPUT_LAYER = self.OUTPUT_LAYER - 1
        self.neuron_outputs = self.create_outputs()
        self.neuron_biases = self.create_biases()
        self.neuron_weights = self.create_weights()
        self.neuron_error_signals = self.create_error_signals()
        self.data_builder = data_builder

        self.timer.network_init_end = self.timer.log_time()


    def random_dec_in_range(self):
        return (random.randint(0, 2000) - 1000)/1000

    def create_outputs(self):
        outputs = []
        for layer in range(0, self.NO_OF_LAYERS):
            newLayer = []
            for neuron in range(0, self.layers[layer]):
                newLayer.append(0)
            outputs.append(newLayer)
        return outputs

    def create_biases(self):
        biases = []
        for layer in range(0, self.NO_OF_LAYERS):
            new_layer = []
            for neuron in range(0, self.layers[layer]):
                new_layer.append(self.random_dec_in_range())
            biases.append(new_layer)

        return biases

    def create_weights(self):
        weights = []
        for layer in range(0, self.NO_OF_LAYERS):
            new_layer = []
            for neuron in range(0, self.layers[layer]):
                neurons = []
                for previous_layer_neurons in range(0, self.layers[layer-1]):
                    neurons.append(self.random_dec_in_range())

                new_layer.append(neurons)
            weights.append(new_layer)
        return weights

    def create_error_signals(self):
        outputs = []
        for layer in range(0, self.NO_OF_LAYERS):
            newLayer = []
            for neuron in range(0, self.layers[layer]):
                newLayer.append(0)
            outputs.append(newLayer)
        return outputs

    def train(self):
        for input_item in range(NETWORK_TRAINING_SET_START, NETWORK_TRAINING_SET_END):
            if input_item%100 == 0:
                print("Image number: " + str(input_item))
            input = self.data_builder.get_image(input_item)
            output = self.data_builder.get_label(input_item)
            self.run_through_network(input)
            self.backprop_error(output)
            self.update_weights()

    def test(self):
        correctGuesses = 0
        for input_item in range(NETWORK_TESTING_SET_START, NETWORK_TESTING_SET_END):
            input = self.data_builder.get_image(input_item)
            output = self.data_builder.get_label(input_item)

            run_through_output = self.run_through_network(input)
            network_guess = run_through_output.index(max(run_through_output))
            actual_answer = output.index(max(output))

            if network_guess == actual_answer:
                correctGuesses = correctGuesses + 1

            if input_item%100 == 0:
                print("Accuracy: " + str(correctGuesses) + "%")
                correctGuesses = 0

    def run_through_network(self, input):
        self.timer.network_run_through_start = self.timer.log_time()
        self.neuron_outputs[0] = input

        for layer in range(1, self.NO_OF_LAYERS):
            for neuron in range(0, self.layers[layer]):
                neuron_bias = self.neuron_biases[layer][neuron]
                neuron_output = 0

                for prev_layer_neuron in range(0, self.layers[layer-1]):
                    prev_layer_neuron_output = self.neuron_outputs[layer-1][prev_layer_neuron]
                    prev_layer_neuron_weight = self.neuron_weights[layer][neuron][prev_layer_neuron]
                    neuron_output += (prev_layer_neuron_output * prev_layer_neuron_weight)
                neuron_output = neuron_output + neuron_bias
                self.neuron_outputs[layer][neuron] = self.sigmoid(neuron_output)

        self.timer.network_run_through_end = self.timer.log_time()
        self.timer.print_network_run_through_time()
        return self.neuron_outputs[self.OUTPUT_LAYER]

    def backprop_error(self, expected_output):
        self.timer.network_backprop_start = self.timer.log_time()

        for output_neuron in range(0, self.layers[self.OUTPUT_LAYER]):
            calculated_output = self.neuron_outputs[self.OUTPUT_LAYER][output_neuron]
            calculated_output_derivitive = self.sigmoid_deriv(calculated_output)
            actual_output = expected_output[output_neuron]
            self.neuron_error_signals[self.OUTPUT_LAYER][output_neuron] = (calculated_output - actual_output) * calculated_output_derivitive

        for layer in range(self.NO_OF_LAYERS-2, 0):
            for neuron in range(0, self.layers[layer]):
                sum = 0
                for next_layer_neuron in range(0, self.layers[layer+1]):
                    next_layer_neuron_error_signal = self.neuron_error_signals[layer+1][next_layer_neuron]
                    next_layer_neuron_weight = self.neuron_weights[layer+1][next_layer_neuron][neuron]
                    sum += next_layer_neuron_weight * next_layer_neuron_error_signal
                self.neuron_error_signals[layer][neuron] = sum * self.sigmoid_deriv(self.neuron_outputs[layer][neuron])

        self.timer.network_backprop_end = self.timer.log_time()
        self.timer.print_network_backprop_time()

    def update_weights(self):
        self.timer.network_update_weights_start = self.timer.log_time()

        for layer in range(1, self.NO_OF_LAYERS):
            for neuron in range(0, self.layers[layer]):
                delta = -0.3 * self.neuron_error_signals[layer][neuron]
                self.neuron_biases[layer][neuron] += delta

                for prev_layer_neuron in range(0, self.layers[layer-1]):
                    prev_layer_neuron_output = self.neuron_outputs[layer-1][prev_layer_neuron]
                    self.neuron_weights[layer][neuron][prev_layer_neuron] += delta * prev_layer_neuron_output

        self.timer.network_update_weights_end = self.timer.log_time()
        self.timer.print_network_update_weights_time()

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)


class mnist_data:

    IMAGE_LOCATION = "C:/Users/Greg/IdeaProjects/NeuralNetwork-Python/resources/trainImage.idx3-ubyte"
    LABEL_LOCATION = "C:/Users/Greg/IdeaProjects/NeuralNetwork-Python/resources/trainLabel.idx1-ubyte"

    images = None
    labels = None

    num_of_images = 0
    num_of_labels = 0

    img_pxl_x = 0
    img_pxl_y = 0

    def __init__(self):
        print("Starting file reader")

    def get_images(self):
        if self.images is None or len(self.images) == 0:
            with open(self.IMAGE_LOCATION, "rb") as idx_images:
                magic_number, self.num_of_images, self.img_pxl_y, self.img_pxl_x = struct.unpack(">IIII", idx_images.read(16))
                all_imgs = np.fromfile(idx_images, dtype=np.uint8)
                all_imgs_2d = np.reshape(all_imgs, (self.num_of_images, self.img_pxl_x*self.img_pxl_y))
            self.images = all_imgs_2d
        return self.images

    def get_labels(self):
        if self.labels is None or len(self.labels) == 0:
            with open(self.LABEL_LOCATION, "rb") as idx_labels:
                magic_number, self.num_of_labels = struct.unpack(">II", idx_labels.read(8))
                all_labels = np.fromfile(idx_labels, np.uint8)
            self.labels = all_labels
        return self.labels


class mnist_batch:

    batch_size: 0

    mnist_data: None
    mnist_images: None
    mnist_labels: None

    def __init__(self, mnist_data, batch_size):
        print("Initializing batch")
        self.mnist_data = mnist_data
        self.mnist_images = mnist_data.get_images()
        self.mnist_labels = mnist_data.get_labels()
        self.batch_size = batch_size

    def get_batch(self):
        print("Getting batch")
        e = self.elimination_matrix(self.mnist_data.num_of_images, self.batch_size)

        image_batch = np.compress(e, self.mnist_images, axis=0)
        label_batch = np.compress(e, self.mnist_labels, axis=0)
        return image_batch, label_batch

    def elimination_matrix(self, size, batch_size):
        e_matrix = np.full(size, False)
        e_matrix[:batch_size] = True
        np.random.shuffle(e_matrix)
        return e_matrix


class network:
    no_of_layers: None
    max_layer_size: None

    layers: None
    biases: None
    weights: None

    def __init__(self, layers):
        print("Creating Network")
        self.layers = layers
        self.no_of_layers = len(layers)
        self.max_layer_size = max(layers)
        self.biases = self.create_biases(layers)
        self.weights = self.create_weights(layers)

    def create_biases(self, layers):
        biases = np.random.random((self.no_of_layers, self.max_layer_size))
        elim_matrix = self.elim_matrix(layers)
        return biases*elim_matrix

    def create_weights(self, layers):
        weights = np.random.random((self.no_of_layers-1, self.max_layer_size, self.max_layer_size))
        elim_matrix = self.weight_elim_matrix(layers)
        results = weights*elim_matrix
        print(results)

    def elim_matrix(self, layers):
        e = np.full((self.no_of_layers, self.max_layer_size), 1)
        for layer in range(0, self.no_of_layers):
            e[layer][layers[layer]:] = 0
        return e

    def weight_elim_matrix(self, layers):
        e = np.array([])
        for layer in range(1, self.no_of_layers):
            e_sub_sub = np.full(self.max_layer_size, 1)
            e_sub_sub[layers[layer-1]:] = 0
            e_sub = np.tile(e_sub_sub, self.max_layer_size)
            e_sub[self.max_layer_size*layers[layer]:] = 0
            e = np.append(e, e_sub)
        e = e.reshape(self.no_of_layers-1, self.max_layer_size, self.max_layer_size)
        return e


print("******************************************************")
print("********************** Starting **********************")
print("******************************************************")

batch_size = 100
layers = Series([784, 70, 35, 10], ["input", "hidden_1", "hidden_2", "output"])

np_start = time.time()

mnist_batch = mnist_batch(mnist_data(), batch_size)
images, labels = mnist_batch.get_batch()

network = network([784, 70, 35, 10])


a = np.full((3, 4, 4), 1)
print(a)
print("---")
e = np.full(4, 1)
e[2:] = 0
e = np.tile(e, 3*4)
print(e)
print("---")

f = a.flatten()
print(f)
print("---")

r = f*e
r = np.reshape(r, (3, 4, 4))
print(r)

np_end = time.time()
print("%.3f s" % (np_end - np_start))


print("******************************************************")
print("**********************   Done   **********************")
print("******************************************************")