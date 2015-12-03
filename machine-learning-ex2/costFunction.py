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

    # =============================================================

    return (J, grad)
