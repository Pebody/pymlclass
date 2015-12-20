# LRCOSTFUNCTION Compute cost and gradient for logistic regression with
# regularization
#    J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
#    theta as the parameter for regularized logistic regression and the
#    gradient of the cost w.r.t. to the parameters.

import numpy as np

from sigmoid import sigmoid


def lrCostFunction(theta, X, y, lmbda):
    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================

    # =============================================================

    return (J, grad)
