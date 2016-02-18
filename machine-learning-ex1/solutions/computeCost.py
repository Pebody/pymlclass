# COMPUTECOST Compute cost for linear regression
#    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#    parameter for linear regression to fit the data points in X and y

import numpy as np


def computeCost(X, y, theta):
    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    # You need to return the following variables correctly
    J = 0

    #  ====================== YOUR CODE HERE ======================

    def h(X, theta):
        return X.dot(theta)

    diff = h(X, theta) - y
    J = diff.T.dot(diff) / (2 * m)

    #  ============================================================

    return J
