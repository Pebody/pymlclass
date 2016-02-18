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

    def h(X, theta):
        return X.dot(theta)

    J = np.float(-y.T * np.nan_to_num(np.log(sigmoid(h(X, theta))).T) -
                 (1 - y).T * np.nan_to_num(np.log(1 - sigmoid(h(X, theta))).T)) / m
    reg_cost = theta.copy()
    reg_cost[0] = 0
    J += (lmbda * reg_cost.T.dot(reg_cost)) / (2 * m)

    grad = np.asarray((sigmoid(h(X, theta)) - y.T).dot(X) / m)[0]
    reg_grad = theta * (float(lmbda) / m)
    reg_grad[0] = 0
    grad += reg_grad

    # =============================================================

    return (J, grad)
