# GRADIENTDESCENT Performs gradient descent to learn theta
# theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
# taking num_iters gradient steps with learning rate alpha

import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):

        # ====================== YOUR CODE HERE ======================

        def h(X, theta):
            return X.dot(theta)
        theta -= alpha * (X.T.dot(h(X, theta) - y) / m)

        # ============================================================

        # Save the cost J in every iteration
        J_history[i] = computeCost(X, y, theta)

    return theta, J_history
