# GRADIENTDESCENTMULTI Performs gradient descent to learn theta
#  theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
#  taking num_iters gradient steps with learning rate alpha

import numpy as np
from computeCostMulti import computeCostMulti


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):

        # ====================== YOUR CODE HERE ======================

        # ============================================================

        # Save the cost J in every iteration
        J_history[i] = computeCostMulti(X, y, theta)

    return theta, J_history
