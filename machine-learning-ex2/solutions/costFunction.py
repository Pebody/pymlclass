# COSTFUNCTION Compute cost and gradient for logistic regression
#    J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
#    parameter for logistic regression and the gradient of the cost
#    w.r.t. to the parameters.

import numpy as np

from sigmoid import sigmoid


def costFunction(theta, X, y):
    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================

    h = lambda X, theta: X.dot(theta)
    J = np.float(-y.T * np.nan_to_num(np.log(sigmoid(h(X, theta))).T) -
                 (1 - y).T * np.nan_to_num(np.log(1 - sigmoid(h(X, theta))).T)) / m
    grad = np.asarray((sigmoid(h(X, theta)) - y.T).dot(X) / m)[0]

    # =============================================================

    return (J, grad)
