# NORMALEQN Computes the closed-form solution to linear regression
#   NORMALEQN(X,y) computes the closed-form solution to linear
#   regression using the normal equations.

import numpy as np


def normalEqn(X, y):
    theta = np.zeros((X.shape[1], 1))

    # ====================== YOUR CODE HERE ======================

    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    # ============================================================

    return theta
