#!/usr/bin/env python3
#
# Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all
#

import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import os

from displayData import displayData
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll


if __name__ == '__main__':
    # Initialization
    os.system('cls' if os.name == 'nt' else 'clear')
    plt.ion()

    # Setup the parameters you will use for this part of the exercise
    input_layer_size = 400  # 20x20 Input Images of Digits
    num_labels = 10         # 10 labels, from 1 to 10
    # (note that we have mapped "0" to label 10)

    # =========== Part 1: Loading and Visualizing Data =============
    #  We start the exercise by first loading and visualizing the dataset.
    #  You will be working with a dataset that contains handwritten digits.
    #

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

    # ============ Part 2: Vectorize Logistic Regression ============
    # In this part of the exercise, you will reuse your logistic regression
    # code from the last exercise. You task here is to make sure that your
    # regularized logistic regression implementation is vectorized. After
    # that, you will implement one-vs-all classification for the handwritten
    # digit dataset.
    #

    print('Training One-vs-All Logistic Regression...')

    lmbda = 0.1
    all_theta = oneVsAll(X, y, num_labels, lmbda)

    input('Program paused. Press enter to continue.')

    # ================ Part 3: Predict for One-Vs-All ================
    # After ...
    pred = predictOneVsAll(all_theta, X)

    print('Training Set Accuracy: %f' % (np.mean((pred == y) * 1) * 100))
