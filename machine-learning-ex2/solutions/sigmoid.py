# SIGMOID Compute sigmoid functoon
#   J = SIGMOID(z) computes the sigmoid of z.

import numpy as np


def sigmoid(z):
    # You need to return the following variables correctly
    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================

    g = 1 / (1 + np.exp(-z))

    # =============================================================

    return g
