import numpy as np


#
# NO OTHER IMPORTS ALLOWED
#


def create_greyscale_histogram(img):
    '''
    returns a histogram of the given image
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    '''
    # TODO
    valueOccurences = np.unique(img, return_counts=True)
    greyscaleVector = np.zeros((256), dtype=int)
    greyscaleVector[valueOccurences[0]] = valueOccurences[1]
    return greyscaleVector


def binarize_threshold(img, t):
    '''
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    '''
    # TODO
    binarizedImg = np.copy(img)
    binarizedImg[img <= t] = 0
    binarizedImg[img > t] = 255
    return binarizedImg


def p_helper(hist, theta: int):
    '''
    Compute p0 and p1 using the histogram and the current theta,
    do not take care of border cases in here
    :param hist:
    :param theta: current theta
    :return: p0, p1
    '''
    # TODO
    p0 = np.sum(hist[0:theta + 1:1])
    p1 = np.sum(hist[theta + 1:len(hist):1])
    return p0, p1


def mu_helper(hist, theta, p0, p1):
    '''
    Compute mu0 and m1
    :param hist: histogram
    :param theta: current theta
    :param p0:
    :param p1:
    :return: mu0, mu1
    '''
    # TODO
    epsilon = np.finfo(float).eps
    theta_vector = np.arange(start=0, stop=256, step=1, dtype=int)
    mu0 = 1 / (p0 + epsilon) * np.sum(np.multiply(theta_vector[0:theta + 1:1], hist[0:theta + 1:1]))
    mu1 = 1 / (p1 + epsilon) * np.sum(np.multiply(theta_vector[theta + 1:len(hist):1], hist[theta + 1:len(hist):1]))
    return mu0, mu1


def calculate_otsu_threshold(hist):
    '''
    calculates theta according to otsus method

    :param hist: 1D array
    :return: threshold (int)
    '''
    # TODO initialize all needed variables
    p0 = np.zeros(256)
    p1 = np.zeros(256)
    mu0 = np.zeros(256)
    mu1 = np.zeros(256)
    variance = np.zeros(256)

    # TODO change the histogram, so that it visualizes the probability distribution of the pixels
    # --> sum(hist) = 1
    normalizedHist = np.divide(hist, np.sum(hist))
    # TODO loop through all possible thetas
    for n in range(255):
        # TODO compute p0 and p1 using the helper function
        p0[n], p1[n] = p_helper(normalizedHist, theta=n)
        # TODO compute mu and m1 using the helper function
        mu0[n], mu1[n] = mu_helper(normalizedHist, theta=n, p0=p0[n], p1=p1[n])
        # TODO compute variance
        totalMeanLevel = p0[n] * mu0[n] + p1[n] * mu1[n]
        variance[n] = p0[n] * np.square((mu0[n] - totalMeanLevel)) + p1[n] * np.square((mu1[n] - totalMeanLevel))
        # TODO update the threshold
    threshold = np.argmax(variance)
    return threshold


def otsu(img):
    '''
    calculates a binarized image using the otsu method.
    Hint: reuse the other methods
    :param image: grayscale image values in range [0, 255]
    :return: np.ndarray binarized image with values {0, 255}
    '''
    # TODO
    greyScaleHistogram = create_greyscale_histogram(img)
    threshold = calculate_otsu_threshold(greyScaleHistogram)
    binarizedImg = binarize_threshold(img, threshold)
    return binarizedImg
