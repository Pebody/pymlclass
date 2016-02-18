import numpy as np
import matplotlib.pyplot as plt

from polyFeatures import polyFeatures


def plotFit(min_x, max_x, mu, sigma, theta, p):
    """
    PLOTFIT Plots a learned polynomial regression fit over an existing figure.
    Also works with linear regression.
    PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
    fit with power p and feature normalization (mu, sigma).
    """

    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.matrix(np.arange(min_x - 15, max_x + 25, 0.05)).T

    # Map the X values
    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly /= sigma

    # Add ones
    X_poly = np.matrix(np.hstack((np.ones((X_poly.shape[0], 1)), X_poly)))

    # Plot
    plt.plot(x, X_poly.dot(theta).T, '-', linewidth=2)
