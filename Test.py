from perceptron import Perceptron
import numpy as np

# The three different functions we test
expected_outputs_AND = np.array([1, 0, 0, 0])
expected_outputs_OR = np.array([1, 1, 1, 0])
expected_outputs_XOR = np.array([0, 1, 1, 0])

# Define function to test the perceptron class on given inputs
def train_perceptron(expected_outputs):
    # Define a list of training inputs
    training_inputs = [np.array([1, 1]), np.array([1, 0]), np.array([0, 1]), np.array([0, 0])]
    perceptron = Perceptron(2)
    perceptron.fit(training_inputs, expected_outputs)

    for training_input in training_inputs:
        prediction = perceptron.predict(training_input)
        print(prediction)

print('AND: ')
train_perceptron(expected_outputs_AND)
print()
print('OR: ')
train_perceptron(expected_outputs_OR)
print()
print('XOR: ')
train_perceptron(expected_outputs_XOR)






