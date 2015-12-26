# PREDICT Predict the label of an input given a trained neural network
#    p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#    trained weights of a neural network (Theta1, Theta2)

import numpy as np

from sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros((m, 1))

    a_1 = np.hstack((np.ones((m, 1)), X))
    a_2 = sigmoid(a_1.dot(Theta1.T))
    a_2 = np.hstack((np.ones((m, 1)), a_2))
    a_3 = sigmoid(a_2.dot(Theta2.T))

    p = np.argmax(a_3, axis=1) + 1

    # =========================================================================

    return p
