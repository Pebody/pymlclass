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

    def h(X, theta):
        return X.dot(theta)

    diff = h(X, theta).T - y
    J = np.float(diff.T.dot(diff) / (2 * m))
    reg_cost = theta.copy()
    reg_cost[0] = 0
    J += (lmbda * reg_cost.T.dot(reg_cost)) / (2 * m)

    grad = X.T.dot(h(X, theta).T - y) / m
    reg_grad = theta * (float(lmbda) / m)
    reg_grad[0] = 0
    grad = grad.A1 + reg_grad

    # =========================================================================

    return (J, grad)
