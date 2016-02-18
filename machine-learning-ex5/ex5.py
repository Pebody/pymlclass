#!/usr/bin/env python3
#
# Machine Learning Online Class - Exercise 5 Regularized Linear Regression and Bias-Variance
#

import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import os

from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from featureNormalize import featureNormalize
from polyFeatures import polyFeatures
from plotFit import plotFit
from validationCurve import validationCurve


if __name__ == '__main__':
    # Initialization
    os.system('cls' if os.name == 'nt' else 'clear')
    plt.ion()

    # =========== Part 1: Loading and Visualizing Data =============
    #  We start the exercise by first loading and visualizing the dataset.
    #  The following code will load the dataset into your environment and plot
    #  the data.
    #

    # Load Training Data
    print('Loading and Visualizing Data ...')

    # Load from ex5data1:
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    data = io.loadmat('ex5data1.mat')
    X, y = np.matrix(data['X']), np.matrix(data['y'])
    Xval, yval = np.matrix(data['Xval']), np.matrix(data['yval'])
    Xtest, ytest = np.matrix(data['Xtest']), np.matrix(data['ytest'])

    # m = Number of examples
    m, n = X.shape

    # Plot training data
    fig = plt.figure()
    plt.plot(X, y, 'rx', linewidth=1.5, markersize=10)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')

    input('Program paused. Press enter to continue.')

    # =========== Part 2: Regularized Linear Regression Cost =============
    #  You should now implement the cost function for regularized linear
    #  regression.
    #

    theta = np.ones((n + 1))
    J, grad = linearRegCostFunction(
        theta, np.hstack((np.ones((m, 1)), X)), y, 1)

    print('Cost at theta = [1, 1]: %f' % (J))
    print('(this value should be about 303.993192)')

    input('Program paused. Press enter to continue.')

    # =========== Part 3: Regularized Linear Regression Gradient =============
    # You should now implement the gradient for regularized linear
    # regression.
    #

    theta = np.ones((n + 1))
    J, grad = linearRegCostFunction(
        theta, np.hstack((np.ones((m, 1)), X)), y, 1)

    print('Gradient at theta = [1, 1]: [%f, %f]' % (grad[0], grad[1]))
    print('(this value should be about [-15.303016, 598.250744])')

    input('Program paused. Press enter to continue.')

    # =========== Part 4: Train Linear Regression =============
    # Once you have implemented the cost and gradient correctly, the
    # trainLinearReg function will use your cost function to train
    # regularized linear regression.
    #
    # Write Up Note: The data is non-linear, so this will not give a great fit.

    #  Train linear regression with lambda = 0
    lmbda = 0
    theta = trainLinearReg(np.hstack((np.ones((m, 1)), X)), y, lmbda)

    #  Plot fit over the data
    plt.plot(X, np.hstack((np.ones((m, 1)), X)).dot(theta).T, '-', linewidth=2)

    input('Program paused. Press enter to continue.')

    # =========== Part 5: Learning Curve for Linear Regression =============
    #  Next, you should implement the learningCurve function.
    #
    #  Write Up Note: Since the model is underfitting the data, we expect to
    #                 see a graph with "high bias" -- slide 8 in ML-advice.pdf
    #

    lmbda = 0
    error_train, error_val = learningCurve(np.hstack((np.ones((m, 1)), X)), y,
                                           np.hstack((np.ones((Xval.shape[0], 1)), Xval)), yval, lmbda)

    fig, ax = plt.subplots()
    ax.plot(np.arange(m), error_train, label='Train')
    ax.plot(np.arange(m), error_val, label='Cross Validation')
    ax.set_title('Learning curve for linear regression')
    ax.set_xlabel('Number of training examples')
    ax.set_ylabel('Error')
    ax.axis([0, 13, 0, 150])
    ax.legend(numpoints=1)

    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m):
        print('  \t%d\t\t%f\t%f' % (i + 1, error_train[i], error_val[i]))

    input('Program paused. Press enter to continue.')

    # =========== Part 6: Feature Mapping for Polynomial Regression ==========
    # One solution to this is to use polynomial regression. You should now
    # complete polyFeatures to map each example into its powers
    #

    p = 8

    # Map X onto Polynomial Features and Normalize
    X_poly = polyFeatures(X, p)
    X_poly, mu, sigma = featureNormalize(X_poly)   # Normalize
    X_poly = np.matrix(np.hstack((np.ones((m, 1)), X_poly)))  # Add Ones

    # Map X_poly_test and normalize (using mu and sigma)
    X_poly_test = polyFeatures(Xtest, p)
    X_poly_test = X_poly_test - mu
    X_poly_test /= sigma
    X_poly_test = np.matrix(np.hstack(
        (np.ones((X_poly_test.shape[0], 1)), X_poly_test)))  # Add Ones

    # Map X_poly_val and normalize (using mu and sigma)
    X_poly_val = polyFeatures(Xval, p)
    X_poly_val = X_poly_val - mu
    X_poly_val /= sigma
    X_poly_val = np.matrix(np.hstack(
        (np.ones((X_poly_val.shape[0], 1)), X_poly_val)))  # Add Ones

    print('Normalized Training Example 1:')
    print(X_poly[1, :])

    input('Program paused. Press enter to continue.')

    # =========== Part 7: Learning Curve for Polynomial Regression ===========
    # Now, you will get to experiment with polynomial regression with multiple
    # values of lmbda. The code below runs polynomial regression with
    # lmbda = 0. You should try running the code with different values of
    # lmbda to see how the fit and learning curve change.
    #

    lmbda = 10
    theta = trainLinearReg(X_poly, y, lmbda)

    # Plot training data and fit
    fig, ax = plt.subplots()
    ax.plot(X, y, 'rx', linewidth=1.5, markersize=10)
    ax.set_xlabel('Change in water level (x)')
    ax.set_ylabel('Water flowing out of the dam (y)')
    ax.set_title('Polynomial Regression Fit (lambda = %f)' % lmbda)
    plotFit(np.min(X), np.max(X), mu, sigma, theta, p)

    fig, ax = plt.subplots()
    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lmbda)
    ax.plot(np.arange(m), error_train, label='Train')
    ax.plot(np.arange(m), error_val, label='Cross Validation')
    ax.set_title('Polynomial Regression Learning Curve (lambda = %f)' % lmbda)
    ax.set_xlabel('Number of training examples')
    ax.set_ylabel('Error')
    ax.axis([0, 13, 0, 150])
    ax.legend(numpoints=1)

    print('\nPolynomial Regression (lambda = %f)' % lmbda)
    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m):
        print('  \t%d\t\t%f\t%f' % (i + 1, error_train[i], error_val[i]))

    input('Program paused. Press enter to continue.')

    # =========== Part 8: Validation for Selecting Lambda =============
    # You will now implement validationCurve to test various values of
    # lmbda on a validation set. You will then use this to select the
    # "best" lambda value.
    #

    lmbda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

    fig, ax = plt.subplots()
    ax.plot(lmbda_vec, error_train, label='Train')
    ax.plot(lmbda_vec, error_val, label='Cross Validation')
    ax.set_xlabel('lambda')
    ax.set_ylabel('Error')
    ax.legend(numpoints=1)

    print('lmbda\t\tTrain Error\tCross Validation Error')
    for i in range(len(lmbda_vec)):
        print('%f\t%f\t%f' % (lmbda_vec[i], error_train[i], error_val[i]))

    input('Program paused. Press enter to continue.')
