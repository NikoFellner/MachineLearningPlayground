'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import matplotlib.pyplot as plt


# do not import more modules!


def polarToKart(shape, r, theta):
    '''
    convert polar coordinates with origin in image center to kartesian
    :param shape: shape of the image
    :param r: radius from image center
    :param theta: angle
    :return: y, x
    '''
    y = shape[0] / 2 + r * np.sin(theta)
    x = shape[1] / 2 + r * np.cos(theta)
    return (y, x)


def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    use the fft to generate a magnitude spectrum and shift it to the image center.
    Hint: This can be done with numpy :)
    :param img:
    :return:
    '''
    # getting the 2D fft result
    freq = np.fft.fft2(img)
    # shift it to the center
    freqShift = np.fft.fftshift(freq)
    # calculating the magnitude
    magnitude = 20 * np.log(np.abs(freqShift))
    magnitude.astype("uint8")
    # thresholding
    magnitude[magnitude < 115] = 0
    magnitude[magnitude >= 115] = 255
    return magnitude


def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum:
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring
    :return: feature vector of k features
    '''
    featureVector = np.zeros(k)
    # iterating for the number of features
    for features in range(k):
        thetaCalc = 0
        # iterating through angles [0 pi] defined by the number of sampling steps
        for angle in range(sampling_steps):
            # iterating through radius starting at k*features + r where r is increasing from 0 to number of features
            for r in range(k + 1):
                rCalc = k * features + r
                y, x = polarToKart(magnitude_spectrum.shape, rCalc, thetaCalc)
                if (y < magnitude_spectrum.shape[0]) & (y >= 0) & (x < magnitude_spectrum.shape[1]) & (x > 0):
                    # sum the magnitude up to get the overall intesity of a ring defining a feature
                    featureVector[features] += magnitude_spectrum[int(y), int(x)]
            thetaCalc += np.pi / (sampling_steps - 1)
    return featureVector


def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    """
    Follow the approach to extract Fan features
    Assume all rays have same length regardless of angle.
    Their length should be set by the smallest feasible ray.
    :param magnitude_spectrum:
    :param k: number of fans-like features to extract
    :param sampling_steps: number of rays to sample from in one fan-like area
    :return: feature vector of length k
    """
    featureVector = np.zeros(k)
    # defining the maximum radius by the size of the minimum of height and width
    rmax = np.min((int(magnitude_spectrum.shape[0]/2),int(magnitude_spectrum.shape[1]/2)))-1
    # iterating through the number of features
    for feature in range(k):
        # getting the lower boundary of the angle
        thetaCalcLowerBoundary = feature * (np.pi / k)
        thetaCalc = thetaCalcLowerBoundary
        # dividing the fan in different lines, adding the intensities along them
        for step in range(sampling_steps):
            for r in range(rmax):
                y, x = polarToKart(magnitude_spectrum.shape, r, thetaCalc)
                featureVector[feature] += magnitude_spectrum[int(y), int(x)]
            thetaCalc = thetaCalcLowerBoundary + (step + 1) * (np.pi / (k * sampling_steps))
    return featureVector


def calcuateFourierParameters(img, k, sampling_steps) -> (np.ndarray, np.ndarray):
    '''
    Extract Features in Fourier space following the paper.
    :param img: input image
    :param k: number of features to extract from each method
    :param sampling_steps: number of samples to accumulate for each feature
    :return: R, T feature vectors of length k
    '''
    # get the magnitude spectrum
    magnitudeSpectrum = calculateMagnitudeSpectrum(img)
    # get ring features
    ringFeatureVector = extractRingFeatures(magnitudeSpectrum, k, sampling_steps)
    # get fan features
    fanFeatureVector = extractFanFeatures(magnitudeSpectrum, k, sampling_steps)
    # return both features
    return (ringFeatureVector, fanFeatureVector)
