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

    h = lambda X, theta: X.dot(theta)

    J = np.float(-y.T * np.nan_to_num(np.log(sigmoid(h(X, theta))).T) -
                 (1 - y).T * np.nan_to_num(np.log(1 - sigmoid(h(X, theta))).T)) / m
    reg_theta = theta.copy()
    reg_theta[0] = 0
    J += (lmbda * reg_theta.T.dot(reg_theta)) / (2 * m)

    grad = np.asarray((sigmoid(h(X, theta)) - y.T).dot(X) / m)[0]
    reg = theta * (float(lmbda) / m)
    reg[0] = 0
    grad += reg

    # =============================================================

    return (J, grad)
