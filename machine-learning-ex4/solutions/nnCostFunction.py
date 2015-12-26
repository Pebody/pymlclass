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
    new_labels = np.zeros((y.shape[0], num_labels))

    for i in range(m):
        new_labels[i, int(y[i]) - 1] = 1

    X = np.hstack((np.ones((m, 1)), X))
    a_2 = sigmoid(X.dot(Theta1.T))
    a_2 = np.hstack((np.ones((m, 1)), a_2))
    a_3 = sigmoid(a_2.dot(Theta2.T))

    J = np.sum(np.multiply(-new_labels, np.nan_to_num(np.log(a_3))) -
               np.multiply(1 - new_labels, np.nan_to_num(np.log(1 - a_3)))) / m

    t1 = Theta1[:, 1:]
    t2 = Theta2[:, 1:]
    J += (lmbda * (np.sum(np.power(t1, 2)) + np.sum(np.power(t2, 2)))) / (2 * m)

    for t in range(m):
        a_1 = X[t, :]
        z_2 = a_1.dot(Theta1.T)
        a_2 = sigmoid(z_2)
        a_2 = np.matrix(np.append([1], a_2))
        z_3 = a_2.dot(Theta2.T)
        a_3 = sigmoid(z_3)

        delta_3 = a_3 - new_labels[t, :]
        delta_2 = np.multiply(delta_3.dot(Theta2[:, 1:]), sigmoidGradient(z_2))

        Theta1_grad += delta_2.T.dot(a_1)
        Theta2_grad += delta_3.T.dot(a_2)

    Theta1_grad /= m
    Theta2_grad /= m

    Theta1_grad[:, 1:] += (lmbda * Theta1[:, 1:]) / m
    Theta2_grad[:, 1:] += (lmbda * Theta2[:, 1:]) / m

    # ============================================================

    # Unroll gradients
    grad = np.hstack((Theta1_grad.flatten(), Theta2_grad.flatten()))

    return J, grad
