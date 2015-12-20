# ONEVSALL trains multiple logistic regression classifiers and returns all
# the classifiers in a matrix all_theta, where the i-th row of all_theta
# corresponds to the classifier for label i
#    [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
#    logisitc regression classifiers and returns each of these classifiers
#    in a matrix all_theta, where the i-th row of all_theta corresponds
#    to the classifier for label i

from scipy import optimize
import numpy as np

from solutions.lrCostFunction import lrCostFunction


def oneVsAll(X, y, num_labels, lmbda):
    # Some useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.hstack((np.ones((m, 1)), X))

    # ====================== YOUR CODE HERE ======================
    # Example Code for scipy.optimize:
    #
    # # Set Initial theta
    # initial_theta = np.zeros(X.shape[1],)
    #
    # # Run scipy.optimize to obtain the optimal theta
    # # This function will return theta and the cost
    # res = optimize.minimize(lrCostFunction, initial_theta, args=(X, (y == (c + 1)) * 1, lmbda),
    #                         method='BFGS', jac=True, options={'maxiter': 50})

    for c in range(num_labels):
        # Set Initial theta
        initial_theta = np.zeros(X.shape[1],)

        # Run scipy.optimize to obtain the optimal theta
        # This function will return theta and the cost
        res = optimize.minimize(lrCostFunction, initial_theta, args=(X, (y == (c + 1)) * 1, lmbda),
                                method='BFGS', jac=True, options={'maxiter': 50})
        theta, cost = res.x, res.fun

        all_theta[c, :] = theta

    # ===========================================================

    return all_theta
