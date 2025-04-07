from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal
import matplotlib.pyplot as plt


# TODO: Test the functions imported in lines 1 and 2 of this file.
linear_signal = createChirpSignal(samplingrate=200, duration=1, freqfrom=1, freqto=10, linear=True)
plt.plot(linear_signal)
plt.show()
exponential_signal = createChirpSignal(samplingrate=200, duration=1, freqfrom=1, freqto=10, linear=False)
plt.plot(exponential_signal)
plt.show()
triangle_signal = createTriangleSignal(samples=200, frequency=2,k_max=10000)
plt.plot(triangle_signal)
plt.show()
square_signal = createSquareSignal(samples=200,frequency=2,k_max=10000)
plt.plot(square_signal)
plt.show()
sawtooth_signal = createSawtoothSignal(samples=200,frequency=2,k_max=10000,amplitude=1)
plt.plot(sawtooth_signal)
plt.show()
