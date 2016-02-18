# TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
# regularization parameter lambda
#    theta = TRAINLINEARREG (X, y, lmbda) trains linear regression using
#    the dataset (X, y) and regularization parameter lambda. Returns the
#    trained parameters theta.
#

import numpy as np
from scipy import optimize

from linearRegCostFunction import linearRegCostFunction


def trainLinearReg(X, y, lmbda):
    # Initialize Theta
    initial_theta = np.zeros(X.shape[1])

    # Optimize
    res = optimize.minimize(linearRegCostFunction, initial_theta, args=(X, y, lmbda),
                            method='BFGS', jac=True, options={'maxiter': 200, 'disp': True})
    theta = res.x

    return theta
