# TDT4137 Assignment 4 - Marianne Hernholm
# 1.12: Implement the perceptron model
import numpy as np

# The perceptron class will include all functions essential to the perceptron-model


class Perceptron(object):

    def __init__(self, n, learning_rate=0.01, epochs=100):
        self.n = n  # n: # inputs, helps define # weights needed
        self.lr = learning_rate     # lr: change in weights in each step of training
        self.epochs = epochs    # epochs: the # times the learning-algorithm gets to iterate
        self.weights = np.zeros(n + 1)  # create empty vec of n+1 zeroes

    # The activation-function returns 1 if a certain threshold is reached
    # the predict function takes inputs(test-samples) and sums the dot product of the inputs
    # and the weights for every input
    # it is an implementation of a step function
    # inputs is the training samples
    def predict(self, inputs):
        summation = (np.dot(inputs, self.weights[1:]) + self.weights[0])
        if summation > 0: activation = 1
        else: activation = 0
        return activation

    # the fit-function is our training function
    # it takes in set of training-inputs and their corresponding expected-outputs
    # Both the training inputs and the expected outputs are np vectors
    # this part is the perceptron-algorithm loop, we loop through the training examples

    def fit(self, training_inputs, expected_outputs):
        for _ in range(self.epochs):    # use the chosen # epochs and iterate
            for inputs, label in zip(training_inputs, expected_outputs): # we zip the training ips/ops to get an iterable object
                prediction = self.predict(inputs)  # we run the prediction function on the training-inputs
                self.weights[1:] += self.lr*(label-prediction)*inputs
                self.weights[0] += self.lr*(label-prediction)
                # implementing the learning rule
                # expected:outputs-prediction is our error
                # multiply the error with the learning rate



