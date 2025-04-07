import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from convo import make_kernel, slow_convolve, kernelFlip, zero_padding


#
# NO MORE MODULES ALLOWED
#


def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
    # TODO
    # creating gauss kernel by given sigma and kernel size
    kernel = make_kernel(ksize, sigma)
    # blurr the image with the gauss kernel and convolve the image with the kernel
    filteredImg = slow_convolve(img_in, kernel)
    filteredImg = filteredImg.astype("int")
    return kernel, filteredImg


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    # TODO
    # sobel filters for x and y direction
    kernelGX = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    kernelGY = kernelFlip(np.array([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]]))
    # convolve with both kernels
    # in slow_convolve the kernel is flipped and zero-padding is used
    gx = slow_convolve(img_in, kernelGX)
    gy = slow_convolve(img_in, kernelGY)
    gx = gx.astype("int")
    gy = gy.astype("int")
    return gx, gy


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    # TODO
    # calculating the elementwise absolute values for each pixel
    g = np.sqrt(np.add(np.square(gx), np.square(gy)))
    g = g.astype("int")
    # calculating the direction of each pixel with respect to the four possible quadrants using arctan2 by
    # delivering first y and second x values
    theta = np.arctan2(gy, gx)
    return g, theta


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    # TODO
    # converting radians in degree and modulo with 180 °, such that the result will be only degrees between 0° and 180 °
    deg = np.fmod(np.multiply(angle, (180 / np.pi)), 180)
    angleDeg = np.copy(deg)
    # if (x >= 157.5) or (x >= 0 & x < 22.5) -> x = 0 °
    angleDeg[angleDeg >= 157.5] = 0
    angleDeg[(angleDeg >= 0) & (angleDeg < 22.5)] = int(0)
    # if (x >= 22.5 & x < 67.5) -> x = 45 °
    angleDeg[(angleDeg >= 22.5) & (angleDeg < 67.5)] = int(45)
    # if (x >= 67.5 & x < 112.5) -> x = 90 °
    angleDeg[(angleDeg >= 67.5) & (angleDeg < 112.5)] = int(90)
    # (x >= 112.5 & x < 157.5) -> x = 135 °
    angleDeg[(angleDeg >= 112.5) & (angleDeg < 157.5)] = int(135)
    # checking if deg is numpy array, if not -> np.asscalar: Convert an array of size 1 to its scalar equivalent.
    if type(deg) != numpy.ndarray:
        angleDeg = int(np.asscalar(angleDeg))
    return angleDeg


def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    # TODO Hint: For 2.3.1 and 2 use the helper method above
    max_sup = np.zeros(g.shape)
    # converting the angles of the resulting sobel filtered image into 0, 45, 90 and 135 degree by choosing
    # the nearest neighbor
    thetaNearestNeighbor = convertAngle(theta)
    thetaNearestNeighbor = thetaNearestNeighbor.astype("int")
    # zero pad the sobel absolute values, such that on each side of the image a column/row is added
    g_padded = zero_padding(g, (3, 3))
    # going through all pixel values and look for the local maxima respecting the gradient direction
    for y in range(g.shape[0]):
        for x in range(g.shape[1]):
            # if gradient direction theta = 0: gradient horizontal
            if thetaNearestNeighbor[y, x] == 0:
                # if the values left and right of the actual value are smaller::
                # local maxima -> save the value in max_sup
                if (g_padded[y + 1, x + 1] >= g_padded[y + 1, x + 1 - 1]) & (
                        g_padded[y + 1, x + 1] >= g_padded[y + 1, x + 1 + 1]):
                    max_sup[y, x] = g[y, x]
            # if gradient direction theta = 45: gradient down-left to up-right
            elif thetaNearestNeighbor[y, x] == 45:
                # if the values down left and up right are smaller than the actual value::
                # local maxima -> save the value in max_sup
                if (g_padded[y + 1, x + 1] >= g_padded[y + 1 + 1, x + 1 - 1]) & (
                        g_padded[y + 1, x + 1] >= g_padded[y + 1 - 1, x + 1 + 1]):
                    max_sup[y, x] = g[y, x]
            # if gradient direction theta = 90: gradient vertical
            elif thetaNearestNeighbor[y, x] == 90:
                # if the values above or under the actual value are smaller::
                # local maxima -> save the value in max_sup
                if (g_padded[y + 1, x + 1] >= g_padded[y + 1 - 1, x + 1]) & (
                        g_padded[y + 1, x + 1] >= g_padded[y + 1 + 1, x + 1]):
                    max_sup[y, x] = g[y, x]
            # if gradient direction theta = 135: gradient up-left to down-right
            elif thetaNearestNeighbor[y, x] == 135:
                # if the values up-left and down-right are smaller than the actual value::
                # local maxima -> save the value in max_sup
                if (g_padded[y + 1, x + 1] >= g_padded[y + 1 + 1, x + 1 + 1]) & (
                        g_padded[y + 1, x + 1] >= g_padded[y + 1 - 1, x + 1 - 1]):
                    max_sup[y, x] = g[y, x]
    return max_sup


def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    # TODO
    thresholdedIMG = np.copy(max_sup)
    # first thresholding the image with the upper and lower boundary,
    # x <= lower boundary = 0
    # lower boundary < x <= upper boundary = 1
    # x > upper boundary = 2
    thresholdedIMG[max_sup <= t_low] = 0
    thresholdedIMG[(max_sup > t_low) & (max_sup <= t_high)] = 1
    thresholdedIMG[max_sup > t_high] = 2
    # for all values above the boundary -> set 255
    thresholdedIMG[thresholdedIMG == 2] = 255

    # look for all pixels with values of 255
    # if any pixel surrounding them is of value 1, set them also to 255
    paddedThresholdedIMG = zero_padding(thresholdedIMG, (3, 3))
    xpos = np.where(paddedThresholdedIMG == 255)[0]
    ypos = np.where(paddedThresholdedIMG == 255)[1]
    pos = np.concatenate((np.reshape(xpos, (len(xpos), 1)), np.reshape(ypos, (len(ypos), 1))), axis=1)
    for position in pos:
        slice = paddedThresholdedIMG[position[0] - 1:position[0] + 2, position[1] - 1:position[1] + 2]
        slice[slice == 1] = 255
        paddedThresholdedIMG[position[0] - 1:position[0] + 2, position[1] - 1:position[1] + 2] = slice
    thresholdedIMGShape = thresholdedIMG.shape
    thresholdedIMG = paddedThresholdedIMG[1:thresholdedIMGShape[0] + 1, 1:thresholdedIMGShape[1] + 1]
    return thresholdedIMG


def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result
