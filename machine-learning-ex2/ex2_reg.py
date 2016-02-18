#!/usr/bin/env python3
#
# Machine Learning Online Class - Exercise 2: Logistic Regression
#

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os

from predict import predict
from plotData import plotData
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg

if __name__ == '__main__':
    # Initialization
    os.system('cls' if os.name == 'nt' else 'clear')
    plt.ion()

    # Load Data
    # The first two columns contains the X values and the third column
    # contains the label (y).

    data = np.asmatrix(np.loadtxt('ex2data2.txt', delimiter=','))
    X, y = data[:, :2], data[:, 2]

    plotData(X, y, ['y = 1', 'y = 0'])

    # Put some labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(numpoints=1)

    plt.show()

    input('Program paused. Press enter to continue.\n')

    # =========== Part 1: Regularized Logistic Regression ============
    # In this part, you are given a dataset with data points that are not
    # linearly separable. However, you would still like to use logistic
    # regression to classify the data points.
    #
    # To do so, you introduce more features to use -- in particular, you add
    # polynomial features to our data matrix (similar to polynomial
    # regression).
    #

    # Add Polynomial Features

    # Note that mapFeature also adds a column of ones for us, so the intercept
    # term is handled
    X = mapFeature(X[:, 0], X[:, 1])

    # Initialize fitting parameters
    initial_theta = np.zeros(X.shape[1],)

    # Set regularization parameter lambda to 1
    lmbda = 1

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    cost, grad = costFunctionReg(initial_theta, X, y, lmbda)

    print('Cost at initial theta (zeros): %f' % cost)

    input('Program paused. Press enter to continue.\n')

    # ============= Part 2: Regularization and Accuracies =============
    # Optional Exercise:
    # In this part, you will get to try different values of lambda and
    # see how regularization affects the decision coundart
    #
    # Try the following values of lambda (0, 1, 10, 100).
    #
    # How does the decision boundary change when you vary lambda? How does
    # the training set accuracy vary?
    #

    # Initialize fitting parameters
    initial_theta = np.zeros(X.shape[1],)

    # Set regularization parameter lambda to 1 (you should vary this)
    lmbda = 1

    # Optimize
    res = optimize.minimize(costFunctionReg, initial_theta, args=(X, y, lmbda),
                            method='BFGS', jac=True, options={'maxiter': 400, 'disp': True})
    theta, cost = res.x, res.fun

    # Plot decision boundary
    u = np.linspace(-1, 1.5, 50)
    z = np.frompyfunc(lambda x1, x2: mapFeature(x1, x2).dot(theta), 2, 1) \
        .outer(u, u)
    z = z.T

    cs = plt.contour(u, u, z, [0], linewidth=2)
    cs.collections[0].set_label('Decision Boundary')
    plt.title('lmbda = %g' % lmbda)

    plt.legend(numpoints=1)

    # Compute accuracy on our training set
    p = predict(theta, X)

    print('Train Accuracy: %f' % np.mean((y.T == p) * 1))

    input('Program paused. Press enter to continue.\n')
