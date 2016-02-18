#!/usr/bin/env python3
#
# Machine Learning Online Class - Exercise 4 Neural Network Learning
#

import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import optimize
import os

from displayData import displayData
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients
from predict import predict


if __name__ == '__main__':
    # Initialization
    os.system('cls' if os.name == 'nt' else 'clear')
    plt.ion()

    # Setup the parameters you will use for this exercise
    input_layer_size = 400   # 20x20 Input Images of Digits
    hidden_layer_size = 25    # 25 hidden units
    num_labels = 10           # 10 labels, from 1 to 10
    # (note that we have mapped "0" to label 10)

    # =========== Part 1: Loading and Visualizing Data =============

    # Load Training Data
    print('Loading and Visualizing Data ...')

    data = io.loadmat('ex4data1.mat')  # training data stored in arrays X, y
    y, X = np.matrix(data['y']), np.matrix(data['X'])
    m, n = X.shape

    # Randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]

    displayData(sel)

    input('Program paused. Press enter to continue.')

    # ================ Part 2: Loading Pameters ================
    # In this part of the exercise, we load some pre-initialized
    # neural network parameters.

    print('Loading Saved Neural Network Parameters ...')

    # Load the weights into variables Theta1 and Theta2
    weights = io.loadmat('ex4weights.mat')
    Theta1, Theta2 = np.matrix(weights['Theta1']), np.matrix(weights['Theta2'])

    # Unroll parameters
    nn_params = np.hstack((Theta1.flatten(), Theta2.flatten()))

    # ================ Part 3: Compute Cost (Feedforward) ================
    print('Feedforward Using Neural Network ...')

    # Weight regularization parameter (we set this to 0 here).
    lmbda = 0

    J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                             num_labels, X, y, lmbda)

    print('Cost at parameters (loaded from ex4weights): %f ' % (J))
    print('(this value should be about 0.287629)')

    input('Program paused. Press enter to continue.')

    # =============== Part 4: Implement Regularization ===============
    print('Checking Cost Function (w/ Regularization) ... ')

    # Weight regularization parameter (we set this to 1 here).
    lmbda = 1

    J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                             num_labels, X, y, lmbda)

    print('Cost at parameters (loaded from ex4weights): %f ' % (J))
    print('(this value should be about 0.383770)')

    input('Program paused. Press enter to continue.')

    # ================ Part 5: Sigmoid Gradient  ================
    # Before you start implementing the neural network, you will first
    # implement the gradient for the sigmoid function. You should complete the
    # code in the sigmoidGradient.py file.

    print('Evaluating sigmoid gradient...')

    g = sigmoidGradient(np.array([1, -0.5, 0, 0.5, 1]))
    print('Sigmoid gradient evaluated at [1, -0.5, 0, 0.5, 1]:')
    print(g)

    print('Program paused. Press enter to continue.')

    # ================ Part 6: Initializing Pameters ================
    # In this part of the exercise, you will be starting to implment a two
    # layer neural network that classifies digits. You will start by
    # implementing a function to initialize the weights of the neural network
    # (randInitializeWeights.m)

    print('Initializing Neural Network Parameters ...')

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

    # Unroll parameters
    initial_nn_params = np.hstack(
        (initial_Theta1.flatten(), initial_Theta2.flatten()))
    print(initial_nn_params.shape)

    # =============== Part 7: Implement Backpropagation ===============
    # Once your cost matches up with ours, you should proceed to implement the
    # backpropagation algorithm for the neural network. You should add to the
    # code you've written in nnCostFunction.m to return the partial
    # derivatives of the parameters.
    #

    print('Checking Backpropagation...')

    #  Check gradients by running checkNNGradients
    checkNNGradients()

    input('Program paused. Press enter to continue.')

    # =============== Part 8: Implement Regularization ===============
    # Once your backpropagation implementation is correct, you should now
    # continue to implement the regularization with the cost and gradient.
    #

    print('Checking Backpropagation (w/ Regularization) ...')

    #  Check gradients by running checkNNGradients
    lmbda = 3
    checkNNGradients(lmbda)

    # Also output the costFunction debugging values
    debug_J, debug_grad = nnCostFunction(
        nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)

    print('Cost at (fixed) debugging parameters (w/ lambda = 10): %f (this value should be about 0.576051)' % (debug_J))

    input('Program paused. Press enter to continue.')

    # =================== Part 8: Training NN ===================
    # You have now implemented all the code necessary to train a neural
    # network. To train your neural network, we will now use "fmincg", which
    # is a function which works similarly to "fminunc". Recall that these
    # advanced optimizers are able to train our cost functions efficiently as
    # long as we provide them with the gradient computations.
    #
    print('Training Neural Network...')

    # After you have completed the assignment, change the MaxIter to a larger
    # value to see how more training helps.
    maxIter = 50

    # You should also try different values of lambda
    lmbda = 1

    # Create "short hand" for the cost function to be minimized
    def costFunc(p):
        return nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)

    # Now, costFunction is a function that takes in only one argument (the
    # neural network parameters)
    # res = optimize.minimize(costFunc, initial_nn_params, method='BFGS',
    res = optimize.minimize(costFunc, initial_nn_params, method='CG',
                            jac=True, options={'maxiter': maxIter, 'disp': True})
    nn_params, cost = res.x, res.fun

    # Obtain Theta1 and Theta2 back from nn_params
    Theta1 = nn_params[:(hidden_layer_size * (input_layer_size + 1))
                       ].reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[hidden_layer_size *
                       (input_layer_size + 1):].reshape((num_labels, hidden_layer_size + 1))

    print('Program paused. Press enter to continue.')

    # == == == == == == == == = Part 9: Visualize Weights == == == == == == == == =
    # You can now "visualize" what the neural network is learning by
    # displaying the hidden units to see what features they are capturing in
    # the data.

    print('Visualizing Neural Network...')

    displayData(Theta1[:, 1:])

    input('Program paused. Press enter to continue.')

    # ================= Part 10: Implement Predict =================
    # After training the neural network, we would like to use it to predict
    # the labels. You will now implement the "predict" function to use the
    # neural network to predict the labels of the training set. This lets
    # you compute the training set accuracy.

    pred = predict(Theta1, Theta2, X)

    print('Training Set Accuracy: %f' % (np.mean((pred == y) * 1) * 100))
