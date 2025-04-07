import numpy as np

def createTriangleSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array (np.ndarray)
    # TODO
    time_vector = np.arange(0,1,1/samples).reshape(1,samples)
    k_vector = np.arange(0,k_max,1).reshape(k_max,1)
    signal = 8/np.power(np.pi,2)*(np.power(-1,k_vector)*(np.sin(2*np.pi*(2*k_vector+1)*frequency*time_vector)/np.power((2*k_vector+1),2)))
    signal = np.sum(signal, axis=0)
    return signal


def createSquareSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array (np.ndarray)
    # TODO
    time_vector = np.arange(0, 1, 1 / samples).reshape(1, samples)
    k_vector = np.arange(1, k_max, 1).reshape(k_max-1, 1)
    signal = 4 / np.pi * (np.sin(2 * np.pi * (2 * k_vector - 1) * frequency * time_vector) / (2 * k_vector - 1))
    signal = np.sum(signal, axis=0)
    return signal


def createSawtoothSignal(samples: int, frequency: int, k_max: int, amplitude: int):
    # returns the signal as 1D-array (np.ndarray)
    # TODO
    time_vector = np.arange(0, 1, 1 / samples).reshape(1, samples)
    k_vector = np.arange(1, k_max, 1).reshape(k_max - 1, 1)
    signal = ((amplitude / np.pi) * (np.sin(2 * np.pi * k_vector * frequency * time_vector) / k_vector))
    signal = amplitude / 2 - np.sum(signal, axis=0)
    return signal
