#!/usr/bin/env python3
#
# Machine Learning Online Class
# Exercise 1: Linear regression with multiple variables
#

import numpy as np
import matplotlib.pyplot as plt
import os

from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn

if __name__ == '__main__':
    # Initialization

    # ================ Part 1: Feature Normalization ================

    # Clear and Close Figures
    os.system('cls' if os.name == 'nt' else 'clear')
    plt.ion()

    print('Loading data ...')

    # Load Data
    data = np.asmatrix(np.loadtxt('ex1data2.txt', delimiter=','))
    X = data[:, :2]
    y = data[:, 2]
    m = y.shape[0]

    # Print out some data points
    print('First 10 examples from the dataset: ')
    for i in range(10):
        print("x = [%i %i], y = %i" % (X[i, 0], X[i, 1], y[i]))

    input('Program paused. Press enter to continue.')

    # Scale features and set them to zero mean
    print('Normalizing Features ...')

    X, mu, sigma = featureNormalize(X)

    # Add intercept term to X
    X = np.hstack((np.ones((m, 1)), X))

    # ================ Part 2: Gradient Descent ================

    # ====================== YOUR CODE HERE ====================

    print('Running gradient descent ...')

    # Choose some alpha value
    alpha = 0.01
    num_iters = 400

    # Init Theta and Run Gradient Descent
    theta = np.zeros((3, 1))
    theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

    # Plot the convergence graph
    plt.plot(range(len(J_history)), J_history, '-b')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()

    # Display gradient descent's result
    print('Theta computed from gradient descent:')
    print(theta)

    # Estimate the price of a 1650 sq-ft, 3 br house
    # ====================== YOUR CODE HERE ======================

    price = 0  # You should change this

    # == == == == == == == == == == == == == == == == == == == ==

    print('Predicted price of a 1650 sq-ft, 3 br house ' +
          '(using gradient descent):\n $%f\n' % (price))

    input('Program paused. Press enter to continue.')

    # ================ Part 3: Normal Equations ================

    print('Solving with normal equations...')

    # ====================== YOUR CODE HERE ====================

    # Load Data
    data = np.asmatrix(np.loadtxt('ex1data2.txt', delimiter=','))
    X = data[:, :2]
    y = data[:, 2]
    m = y.shape[0]

    # Add intercept term to X
    X = np.hstack((np.ones((m, 1)), X))

    # Calculate the parameters from the normal equation
    theta = normalEqn(X, y)

    # Display normal equation's result
    print('Theta computed from the normal equations:')
    print(theta)

    # Estimate the price of a 1650 sq-ft, 3 br house
    # ====================== YOUR CODE HERE ======================

    price = 0  # You should change this

    # ============================================================

    print('Predicted price of a 1650 sq-ft, 3 br house ' +
          '(using normal equations):\n $%f\n' % (price))
