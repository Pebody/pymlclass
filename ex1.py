#!/usr/bin/env python3
#
# Machine Learning Online Class - Exercise 1: Linear Regression
#

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import os

from warmUpExercise import warmUpExercise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent


if __name__ == '__main__':
    # Initialization
    os.system('cls' if os.name == 'nt' else 'clear')
    plt.ion()

    # ==================== Part 1: Basic Function ====================
    # Complete warmUpExercise.py
    print('Running warmUpExercise ...')
    print('5x5 Identity Matrix: ')
    print(warmUpExercise())

    input('Program paused. Press enter to continue.')

    # ======================= Part 2: Plotting =======================
    print('Plotting Data ...')
    data = np.asmatrix(np.loadtxt('ex1data1.txt', delimiter=','))
    X, y = data[:, 0], data[:, 1]
    m = y.shape[0]  # number of training examples

    # Plot Data
    # Note: You have to complete the code in plotData.py
    plotData(X, y)

    input('Program paused. Press enter to continue.')

    # =================== Part 3: Gradient descent ===================
    print('Running Gradient Descent ...')

    X = np.hstack((np.ones((m, 1)), data[:, 0]))  # Add a column of ones to x
    theta = np.zeros((2, 1))  # initialize fitting parameters

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    # compute and display initial cost
    print(computeCost(X, y, theta))

    # run gradient descent
    theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

    # print theta to screen
    print('Theta found by gradient descent: ' + '%f %f' % (theta[0], theta[1]))

    # Plot the linear fit
    plt.plot(X[:, 1], X * theta, '-', label="Linear regression")
    plt.legend()
    plt.show()

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.matrix([1, 3.5]) * theta
    print('For population = 35,000, we predict a profit of %f' %
          (float(predict1) * 10000))
    predict2 = np.matrix([1, 7]) * theta
    print('For population = 70,000, we predict a profit of %f' %
          (float(predict2) * 10000))

    input('Program paused. Press enter to continue.')

    # ============= Part 4: Visualizing J(theta_0, theta_1) =============
    print('Visualizing J(theta_0, theta_1) ...')

    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    xv, yv = np.meshgrid(theta0_vals, theta1_vals)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

    # Fill out J_vals
    for i in range(theta0_vals.shape[0]):
        for j in range(theta1_vals.shape[0]):
            t = np.matrix([theta0_vals[i], theta1_vals[j]]).T
            J_vals[i, j] = computeCost(X, y, t)

    # Surface plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xv, yv, J_vals, cmap=cm.coolwarm)
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.show()

    # Contour plot
    # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    fig = plt.figure()
    plt.contourf(xv, yv, J_vals.T, np.logspace(-2, 3, 20), cmap=cm.coolwarm)
    plt.plot(theta[0], theta[1], 'rx', markersize=13)
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.show(block=True)
