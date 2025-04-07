import numpy as np
import matplotlib.pyplot as plt

# Complete this function such that returns the clipped values of the 1D
# numpy array passed as input to the given minimum and maximum. Example:
#    1 2 3 4 5 6 7 8
# clipped to a minimum of 3 and a maximum of 6 should give
#    3 3 3 4 5 6 6 6
# Note that the input array must not be modified.
def clip(array, minimum, maximum):
    x = np.copy(array)
    x[x<minimum] = minimum
    x[x>maximum] = maximum

    # min_array = np.where(array<minimum, minimum, array)
    # max_array = np.where(min_array>maximum, maximum, min_array)
    # x = max_array
    return x


if __name__ == '__main__':
    array = np.random.rand(100)
    result = clip(array, 0.2, 0.8)
    plt.plot(array, result, '.')
    plt.show()
