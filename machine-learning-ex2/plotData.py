# PLOTDATA Plots the data points X and y into a new figure
#   PLOTDATA(x,y) plots the data points with + for the positive examples
#   and o for the negative examples. X is assumed to be a Mx2 matrix.

import matplotlib.pyplot as plt
import numpy as np


def plotData(X, y, labels):

    # ====================== YOUR CODE HERE ======================

    # Create New Figure
    fig, ax = plt.subplots()

    # Find Indices of Positive and Negative Examples
    idx1, idx2 = np.where(y == 1)[0], np.where(y == 0)[0]

    # Plot Examples
    ax.plot(X[idx1, 0], X[idx1, 1], 'k+', label=labels[0],
            linewidth=2, markersize=7)
    ax.plot(X[idx2, 0], X[idx2, 1], 'ko', label=labels[1],
            c='y', markersize=7)

    # ============================================================
