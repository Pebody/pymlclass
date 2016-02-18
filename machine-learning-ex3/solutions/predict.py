# PREDICT Predict the label of an input given a trained neural network
#    p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#    trained weights of a neural network (Theta1, Theta2)

import numpy as np

from sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    # Useful values
    m, n = X.shape
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros((m, 1))

    # ====================== YOUR CODE HERE ======================

    def h(X, theta):
        return X.dot(theta)

    X = np.hstack((np.ones((m, 1)), X))

    a2 = sigmoid(h(X, Theta1.T))
    a2 = np.hstack((np.ones((m, 1)), a2))
    a3 = sigmoid(h(a2, Theta2.T))

    p = np.argmax(a3, axis=1) + 1

    # =============================================================

    return p
