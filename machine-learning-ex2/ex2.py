#!/usr/bin/env python3
#
# Machine Learning Online Class - Exercise 2: Logistic Regression
#

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os

from plotData import plotData
from costFunction import costFunction
from sigmoid import sigmoid
from predict import predict

if __name__ == '__main__':
    # Initialization
    os.system('cls' if os.name == 'nt' else 'clear')
    plt.ion()

    # Load Data
    # The first two columns contains the exam scores and the third column
    # contains the label.
    data = np.asmatrix(np.loadtxt('ex2data1.txt', delimiter=','))
    X, y = data[:, :2], data[:, 2]

    # ==================== Part 1: Plotting ====================
    #  We start the exercise by first plotting the data to understand the
    #  the problem we are working with.

    print('Plotting data with + indicating (y = 1) examples and o ' +
          'indicating (y = 0) examples.')

    plotData(X, y, ['Admitted', 'Not admitted'])

    # Put some labels and Legend
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(numpoints=1)

    plt.show()

    input('Program paused. Press enter to continue.\n')

    # ============ Part 2: Compute Cost and Gradient ============
    #  In this part of the exercise, you will implement the cost and gradient
    #  for logistic regression. You neeed to complete the code in
    #  costFunction.m

    #  Setup the data matrix appropriately, and add ones for the intercept term
    m, n = X.shape

    # Add intercept term to x and X_test
    X = np.hstack((np.ones((m, 1)), X))

    # Initialize fitting parameters
    initial_theta = np.zeros((n + 1))

    # Compute and display initial cost and gradient
    cost, grad = costFunction(initial_theta, X, y)

    print('Cost at initial theta (zeros): %f' % (cost))
    print('Gradient at initial theta (zeros):')
    print(grad)

    input('Program paused. Press enter to continue.\n')

    # ============= Part 3: Optimizing using fminunc  =============
    res = optimize.minimize(costFunction, initial_theta, args=(X, y),
                            method='BFGS', jac=True, options={'maxiter': 400})
    theta, cost = res.x, res.fun

    # Print theta to screen
    print('Cost at theta found by fminunc: %f' % cost)
    print('theta:')
    print(theta)

    # Plot decision boundary
    plot_x = np.array([min(X[:, 1]) - 2,  max(X[:, 1]) + 2])
    plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
    plt.plot(plot_x.reshape(2,), plot_y.reshape(2,), label='Decision Boundary')
    plt.legend(numpoints=1)

    input('Program paused. Press enter to continue.\n')

    # ============== Part 4: Predict and Accuracies ==============
    # After learning the parameters, you'll like to use it to predict the outcomes
    # on unseen data. In this part, you will use the logistic regression model
    # to predict the probability that a student with score 45 on exam 1 and
    # score 85 on exam 2 will be admitted.
    #
    # Furthermore, you will compute the training and test set accuracies of
    # our model.
    #
    # Your task is to complete the code in predict.m

    # Predict probability for a student with score 45 on exam 1
    # and score 85 on exam 2

    prob = sigmoid(np.array([1, 45, 85]).dot(theta))
    print('For a student with scores 45 and 85, we predict an admission ' +
          'probability of %f' % prob)

    # Compute accuracy on our training set
    p = predict(theta, X)

    acc = np.mean((y.T == ((sigmoid(X.dot(theta)) >= 0.5) * 1)) * 1)
    print('Train Accuracy: %f' % acc)

    input('Program paused. Press enter to continue.\n')
