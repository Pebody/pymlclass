# DISPLAYDATA Display 2D data in a nice grid
#    h, display_array = DISPLAYDATA(X, example_width) displays 2D data
#    stored in X in a nice grid. It returns the figure handle h and the
#    displayed array if requested.

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from math import sqrt, floor, ceil


def displayData(X, example_width=None):
    # Set example_width automatically if not passed in
    if not example_width or example_width is None:
        example_width = floor(sqrt(X.shape[1]))

    # Compute rows, cols
    m, n = X.shape
    example_height = floor(n / example_width)

    # Compute number of items to display
    display_rows = floor(sqrt(m))
    display_cols = ceil(m / display_rows)

    # Setup blank display
    display_array = - np.ones((display_rows * example_height, display_cols * example_width))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break
            # Copy the patch

            # Get the max value of the patch
            max_val = np.max(np.absolute(X[curr_ex, :]))

            rows = np.array([x + j * example_height for x in range(example_height)], dtype=np.intp)
            columns = np.array([x + i * example_width for x in range(example_width)], dtype=np.intp)
            display_array[rows[:, np.newaxis], columns] = np.reshape(
                X[curr_ex, :], (example_height, example_width)).copy() / max_val

            curr_ex += 1
        if curr_ex > m:
            break

    # Display gray-scaled Image
    # h = plt.imshow(np.rot90(display_array, 3), cmap=cm.gray)
    display_array = np.rot90(np.fliplr(display_array))
    h = plt.imshow(display_array, cmap=cm.gray)

    # Do not show axis
    plt.axis('off')
    plt.show()

    return h, display_array
