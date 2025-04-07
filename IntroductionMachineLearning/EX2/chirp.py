import numpy as np

def createChirpSignal(samplingrate: int, duration: int, freqfrom: int, freqto: int, linear: bool):
    # returns the chirp signal as list or 1D-array
    # TODO
    t = np.arange(0, duration, 1 / samplingrate)
    if linear:
        linear_chirp_rate = (freqto-freqfrom)/duration
        chirp_signal = np.sin(2*np.pi*(freqfrom+linear_chirp_rate/2*t)*t)
    else:
        exponential_chirp_rate = np.power(freqto,(1/duration))/freqfrom
        chirp_signal = np.sin(2*np.pi*freqfrom/np.log(exponential_chirp_rate)*(np.power(exponential_chirp_rate,t)-1))

    return chirp_signal
