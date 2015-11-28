# plotData Plots the data points x and y into a new figure
# plotData(x,y) plots the data points and gives the figure axes labels of
# population and profit.

import matplotlib.pyplot as plt


def plotData(X, y):
    fig = plt.figure()

    # ====================== YOUR CODE HERE ======================

    plt.plot(X, y, 'rx', label='Training data')
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.show()

    # ============================================================
