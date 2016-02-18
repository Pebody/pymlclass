# LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
# regression with multiple variables
#   J, grad = LINEARREGCOSTFUNCTION(X, y, theta, lmbda) computes the
#   cost of using theta as the parameter for linear regression to fit the
#   data points in X and y. Returns the cost in J and the gradient in grad

import numpy as np


def linearRegCostFunction(theta, X, y, lmbda):
    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost and gradient of regularized linear
    #               regression for a particular choice of theta.
    #
    #               You should set J to the cost and grad to the gradient.
    #

    # =========================================================================

    return (J, grad)
