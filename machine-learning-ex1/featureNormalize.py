# FEATURENORMALIZE Normalizes the features in X
#   FEATURENORMALIZE(X) returns a normalized version of X where
#   the mean value of each feature is 0 and the standard deviation
#   is 1. This is often a good preprocessing step to do when
#   working with learning algorithms.

import numpy as np


def featureNormalize(X):
    # You need to set these values correctly
    X_norm = X[:, :]
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))

    # == == == == == == == == == YOUR CODE HERE == == == == == == == == ==

    # == == == == == == == == == == == == == == == == == == == == == == ==

    return (X_norm, mu, sigma)
