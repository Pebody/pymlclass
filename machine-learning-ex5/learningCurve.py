# LEARNINGCURVE Generates the train and cross validation set errors needed
# to plot a learning curve
#    error_train, error_val = ...
#        LEARNINGCURVE(X, y, Xval, yval, lmbda) returns the train and
#        cross validation set errors for a learning curve. In particular,
#        it returns two vectors of the same length - error_train and
#        error_val. Then, error_train[i] contains the training error for
#        i examples (and similarly for error_val[i]).
#
#    In this function, you will compute the train and test errors for
#    dataset sizes from 1 up to m. In practice, when working with larger
#    datasets, you might want to do this in larger intervals.
#
import numpy as np

from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg


def learningCurve(X, y, Xval, yval, lmbda):
    # Number of training examples
    m = X.shape[0]

    # You need to return these values correctly
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return training errors in
    #               error_train and the cross validation errors in error_val.
    #               i.e., error_train[i] and
    #               error_val[i] should give you the errors
    #               obtained after training on i examples.
    #
    # Note: You should evaluate the training error on the first i training
    #       examples (i.e., X[:i+1, :] and y[:i+1]).
    #
    #       For the cross-validation error, you should instead evaluate on
    #       the _entire_ cross validation set (Xval and yval).
    #
    # Note: If you are using your cost function (linearRegCostFunction)
    #       to compute the training and cross validation error, you should
    #       call the function with the lmbda argument set to 0.
    #       Do note that you will still need to use lmbda when running
    #       the training to obtain the theta parameters.
    #
    # Hint: You can loop over the examples with the following:
    #
    #      for i in range(m):
    #           % Compute train/cross validation errors using training examples
    #           % X[:i+1, :] and y[:i+1], storing the result in
    #           % error_train[i] and error_val[i]
    #           ....
    #
    #

    # ---------------------- Sample Solution ----------------------

    # -------------------------------------------------------------

    # =============================================================

    return error_train, error_val
