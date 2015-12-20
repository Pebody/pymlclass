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

    # =============================================================

    return p
