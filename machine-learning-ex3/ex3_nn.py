#!/usr/bin/env python3
#
# Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks
#

import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import os

from displayData import displayData
from predict import predict


if __name__ == '__main__':
    # Initialization
    os.system('cls' if os.name == 'nt' else 'clear')
    plt.ion()

    # Setup the parameters you will use for this exercise
    input_layer_size = 400  # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10          # 10 labels, from 1 to 10
    # (note that we have mapped "0" to label 10)

    # =========== Part 1: Loading and Visualizing Data =============

    # Load Training Data
    print('Loading and Visualizing Data ...')

    data = io.loadmat('ex3data1.mat')  # training data stored in arrays X, y
    y, X = np.matrix(data['y']), np.matrix(data['X'])
    m, n = X.shape

    # Randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]

    displayData(sel)

    input('Program paused. Press enter to continue.')

    # ================ Part 2: Loading Pameters ================
    # In this part of the exercise, we load some pre-initialized
    # neural network parameters.

    print('Loading Saved Neural Network Parameters ...')

    # Load the weights into variables Theta1 and Theta2
    weights = io.loadmat('ex3weights.mat')
    Theta1, Theta2 = np.matrix(weights['Theta1']), np.matrix(weights['Theta2'])

    # ================= Part 3: Implement Predict =================
    # After training the neural network, we would like to use it to predict
    # the labels. You will now implement the "predict" function to use the
    # neural network to predict the labels of the training set. This lets
    # you compute the training set accuracy.

    pred = predict(Theta1, Theta2, X)

    print('Training Set Accuracy: %f' % (np.mean((pred == y) * 1) * 100))

    input('Program paused. Press enter to continue.')

    # To give you an idea of the network's output, you can also run
    # through the examples one at the a time to see what it is predicting.

    # Randomly permute examples
    rp = np.random.permutation(m)

    for i in range(m):
        # Display
        print('Displaying Example Image')
        displayData(X[rp[i], :])

        pred = predict(Theta1, Theta2, X[rp[i], :])
        print('Neural Network Prediction: %d (digit %d)' % (pred, pred % 10))

        # Pause
        input('Program paused. Press enter to continue.')
