import random
import math
import time

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


class MnistFileReader:

    timer = None

    IMAGE_LOCATION = "C:/Users/Greg/IdeaProjects/NeuralNetwork-Python/resources/trainImage.idx3-ubyte"
    LABEL_LOCATION = "C:/Users/Greg/IdeaProjects/NeuralNetwork-Python/resources/trainLabel.idx1-ubyte"

    cached_images = []
    cached_labels = []

    def __init__(self, timer):
        print("new file reader")
        self.timer = timer

    def read_images(self):
        if len(self.cached_images) == 0:
            content = []
            with open(self.IMAGE_LOCATION, "rb") as idx_file:
                content = idx_file.read()
            content = content[16:]
            self.cached_images = content
        return self.cached_images

    def get_image(self, index):
        self.timer.get_image_start = timer.log_time()

        all_images = self.read_images()
        image_at_index = all_images[index*IMAGE_SIZE_BYTES:(index+1)*IMAGE_SIZE_BYTES]
        image_as_input = []
        for bte in image_at_index:
            image_as_input.append(bte/256)

        self.timer.get_image_end = timer.log_time()
        self.timer.print_get_image_time()
        return image_as_input

    def read_labels(self):
        if len(self.cached_labels) == 0:
            content = []
            with open(self.LABEL_LOCATION, "rb") as idx_file:
                content = idx_file.read()

            content = content[8:]
            self.cached_labels = content
        return self.cached_labels

    def get_label(self, index):
        self.timer.get_label_start = timer.log_time()

        all_labels = self.read_labels()
        output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        output[all_labels[index]] = 1

        self.timer.get_label_end = timer.log_time()
        self.timer.print_get_label_time()
        return output


class Timer:

    print_data = 0
    print_network = 1

    program_start = 0
    program_end = 0

    get_image_start = 0
    get_image_end = 0

    get_label_start = 0
    get_label_end = 0

    network_init_start = 0
    network_init_end = 0

    network_run_through_start = 0
    network_run_through_end = 0

    network_backprop_start = 0
    network_backprop_end = 0

    network_update_weights_start = 0
    network_update_weights_end = 0

    def log_time(self):
        return time.time()

    def print_get_image_time(self):
        if self.print_data == 1:
            print("GETTING IMAGE: " + str(self.get_image_end - self.get_image_start))

    def print_get_label_time(self):
        if self.print_data == 1:
            print("GETTING LABEL: " + str(self.get_label_end - self.get_label_start))

    def print_network_init_time(self):
        if self.print_network == 1:
            print("NETWORK INIT: " + str(self.network_init_end - self.network_init_start))

    def print_network_run_through_time(self):
        if self.print_network == 1:
            print("NETWORK RUN THROUGH: " + str(self.network_run_through_end - self.network_run_through_start))

    def print_network_backprop_time(self):
        if self.print_network == 1:
            print("NETWORK BACK PROP: " + str(self.network_backprop_end - self.network_backprop_start))

    def print_network_update_weights_time(self):
        if self.print_network == 1:
            print("NETWORK UPDATE WEIGHTS: " + str(self.network_update_weights_end - self.network_update_weights_start))

print("******************************************************")
print("********************** Starting **********************")
print("******************************************************")

timer = Timer()
timer.program_start = timer.log_time()

file_reader = MnistFileReader(timer)

network = Network(NETWORK_LAYERS, file_reader, timer)
network.train()
network.test()

timer.program_end = timer.log_time()
