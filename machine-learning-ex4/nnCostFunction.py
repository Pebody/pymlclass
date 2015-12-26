# NNCOSTFUNCTION Implements the neural network cost function for a two layer
# neural network which performs classification
#    J, grad = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
#    X, y, lambda) computes the cost and gradient of the neural network. The
#    parameters for the neural network are "unrolled" into the vector
#    nn_params and need to be converted back into the weight matrices.
#
#    The returned parameter grad should be a "unrolled" vector of the
#    partial derivatives of the neural network.
#

import numpy as np

from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    if nn_params.shape[0] != 1:
        nn_params = nn_params.reshape((1, nn_params.shape[0]))

    Theta1 = nn_params[:, :(hidden_layer_size * (input_layer_size + 1))
                       ].reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[:, hidden_layer_size *
                       (input_layer_size + 1):].reshape((num_labels, hidden_layer_size + 1))

    # Setup some useful variables
    m = X.shape[0]

    # You need to return the following variables correctly
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # ====================== YOUR CODE HERE ======================

    # ============================================================

    # Unroll gradients
    grad = np.hstack((Theta1_grad.flatten(), Theta2_grad.flatten()))

    return J, grad
