'''
Created on 05.10.2016
Modified on 23.12.2020

@author: Daniel
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.signal


# do not import more modules!


def drawCircle(img, x, y):
    '''
    Draw a circle at circle of radius 5px at (x, y) stroke 2 px
    This helps you to visually check your methods.
    :param img: a 2d nd-array
    :param y:
    :param x:
    :return: img with circle at desired position
    '''
    cv.circle(img, (x, y), 5, 255, 2)
    return img


def drawLine(img, yLeft, yRight):
    '''
    Draw a line with thickness 2px for given start y positon and end y position at the edges of the image
    :param img: a 2d nd-array
    :param x:
    :param y:
    :return: img with line for given locations y1 and y2
    '''
    startPoint = (0, yLeft)
    endPoint = (img.shape[1], yRight)
    cv.line(img, startPoint, endPoint, 255, 2)
    return img


def drawLineK1K3(img, k1, k3):
    '''
    Draw a line in a image given two points
    :param img: input image
    :param k1: startpunkt
    :param k3: endpunkt
    :return: input image with line between k1 and k3
    '''
    cv.line(img, (k1[1], k1[0]), (k3[1], k3[0]), 255, 1)
    return img


def make_kernel(ksize, sigma):
    '''
    create Gauss Kernel
    :param ksize: kernel size
    :param sigma:
    :return: gauss filter
    '''
    x = np.reshape(np.abs(np.arange(int(ksize / 2), -int(ksize / 2) - 1, -1)), (1, ksize))
    y = np.transpose(x)
    kernel = (1 / (2 * np.pi * np.square(sigma))) * np.exp(
        -1 * np.divide((np.square(x) + np.square(y)), np.multiply(2, np.square(sigma))))
    kernelSum = np.sum(np.sum(kernel, axis=1), axis=0)
    kernelNormalized = np.divide(kernel, kernelSum)
    return kernelNormalized  # implement the Gaussian kernel here


def binarizeAndSmooth(img) -> np.ndarray:
    '''
    First Binarize using threshold of 115, then smooth with gauss kernel (5, 5)
    :param img: greyscale image in range [0, 255]
    :return: preprocessed image
    '''
    # binarize with threshold 115
    threshold = 115
    binarizedIMG = np.copy(img)
    binarizedIMG[img <= threshold] = 0
    binarizedIMG[img > threshold] = 255
    # smooth image with gaussian blur
    smoothedIMG = cv.GaussianBlur(binarizedIMG, (5, 5), 1)
    return smoothedIMG


def drawLargestContour(img) -> np.ndarray:
    '''
    find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
    :param img: preprocessed image (mostly b&w)
    :return: contour image
    '''
    img = img.astype("uint8")
    contourIMG = np.zeros(img.shape)
    # find contours in image
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # sort contours
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
    # use the first (largest) contour
    largestContour = sorted_contours[0]
    #draw contour
    contourIMG = cv.drawContours(contourIMG, largestContour, -1, (255,0,0),2)
    return contourIMG


def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    '''
    Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
    (For help check Palmprint_Algnment_Helper.pdf section 2b)
    :param contour_img:
    :param x: position of the image column to run along
    :return: y-values in np.ndarray in shape (6,)
    '''
    # just use the column specified by x
    contourColumn = contour_img[:, x]
    # find all points in this column with intensity
    idx = np.where(contourColumn > 0)
    # get the difference of idx, to locate intensity gaps
    upperEdges = np.diff(idx)
    upperEdges = upperEdges.reshape(upperEdges.shape[1])
    # if the difference >1 there is a gap between the contour lines
    idxIntersections = np.where(upperEdges > 1)
    # getting the real image index by inherit the found gaps in the intensitiy, since the first one is the gap on the border, use the second
    intersections = idx[0][idxIntersections[0][1:]]
    # thickness of contour is 5 px, centereing the intersection by subtracting 3 px
    intersectionsCentered = np.subtract(intersections, 3)
    # if more intersections detected, just use the first 6
    if intersectionsCentered.shape[0] > 6:
        intersectionsCentered = intersectionsCentered[0:6:1]
    return intersectionsCentered


def findKPoints(img, y1, x1, y2, x2) -> tuple:
    '''
    given two points and the contour image, find the intersection point k
    :param img: binarized contour image (255 == contour)
    :param y1: y-coordinate of point
    :param x1: x-coordinate of point
    :param y2: y-coordinate of point
    :param x2: x-coordinate of point
    :return: intersection point k as a tuple (ky, kx)
    '''
    # getting gradient of two given points (x1, y1) and (x2, y2)
    lineGradients = np.divide(np.subtract(y2, y1), np.subtract(x2, x1))
    # defining start and endpoints for drawing a line and then finding the intersection of the contour and the drawn line
    startPoints = np.subtract(y1, np.multiply(lineGradients, x1)).astype("uint8")
    endPoints = np.add(y1, np.multiply(lineGradients, np.subtract(img.shape[1], x2))).astype("uint8")
    # helper image for the previous defined line
    lineIMG = np.zeros(img.shape)
    lineIMG = drawLine(lineIMG, startPoints, endPoints)
    # iterating through all columns to find an intersection of the contour and the line
    for column in range(img.shape[1]):
        contLineColumn = lineIMG[:, column]
        imgColumn = img[:, column]
        diff = np.add(contLineColumn, imgColumn)
        # intersection defined where the intensity sums up to 510
        intersection = np.where(diff == 510)
        # if intersection found, stop iterating throught the image and save the found y-position and x-position in k
        if len(intersection[0]) != 0:
            k = (intersection[0][0], column)
            break
    return k


def getCoordinateTransform(k1, k2, k3) -> np.ndarray:
    '''
    Get a transform matrix to map points from old to new coordinate system defined by k1-3
    Hint: Use cv2 for this.
    :param k1: point in (y, x) order
    :param k2: point in (y, x) order
    :param k3: point in (y, x) order
    :return: 2x3 matrix rotation around origin by angle
    '''
    # gradient of line K1K3
    gradientK1K3 = np.subtract(k1[0], k3[0]) / np.subtract(k1[1], k3[1])
    helperIMG = np.zeros((np.max((k1[0], k2[0], k3[0])) + 10, np.max((k1[1], k2[1], k3[1])) + 10))
    # creating helper image just containing a drawn line between points k1 and k3
    helperIMGK1K3 = np.copy(helperIMG)
    helperIMGK1K3 = drawLineK1K3(helperIMGK1K3, k1, k3)

    # getting the perpendicular gradient
    gradientPerpendicular = -1 / gradientK1K3
    # with the calculated gradient, get the endpoint of the line defined by k2 and the perpendicular intersection with line k1k3
    k2LineYENDPOINT = k2[0] + gradientPerpendicular * (helperIMG.shape[1] - k2[1])
    k2ENDPOINT = (int(k2LineYENDPOINT), helperIMG.shape[1])
    # creating an helper image with just the line defined by the k2 points
    helperIMGK2 = np.copy(helperIMG)
    helperIMGK2 = drawLineK1K3(helperIMGK2, k2, k2ENDPOINT)

    # find the new origin by iterating thorugh the columns of the helper images and find the intersection of them
    for column in range(helperIMG.shape[1] - k2[1]):
        helperIMGK1K3COLUMN = helperIMGK1K3[:, column + k2[1]]
        helperIMGK2COLUMN = helperIMGK2[:, column + k2[1]]
        diff = np.add(helperIMGK1K3COLUMN, helperIMGK2COLUMN)
        # if intersection found, the intensity of both sums up to 510
        intersection = np.where(diff == 510)
        # if intersection found, stop iterating and get the new origin
        if len(intersection[0]) != 0:
            kNEWORIGIN = (intersection[0][0], column + k2[1])
            break
    # getting the absolute position of k1 regarding the new origin
    k1Shiftet = (k1[0] - kNEWORIGIN[0], k1[1] - kNEWORIGIN[1])
    # getting the actual angle for the rotation (+pi to get the rotation from old to new and *-1 for the direction)
    rotAngle = -1*np.rad2deg(np.arctan2(k1Shiftet[1], k1Shiftet[0])+np.pi)
    # creating the rotation matrix
    M = cv.getRotationMatrix2D([int(kNEWORIGIN[0]),int(kNEWORIGIN[1])], rotAngle, 1)
    return M


def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''
    # TODO threshold and blur
    binarizedSmoothedIMG = binarizeAndSmooth(img)
    # TODO find and draw largest contour in image
    largestContourIMG = drawLargestContour(binarizedSmoothedIMG)
    # TODO choose two suitable columns and find 6 intersections with the finger's contour
    x1 = 10
    x2 = 20
    y1 = getFingerContourIntersections(largestContourIMG, x1)
    y2 = getFingerContourIntersections(largestContourIMG, x2)
    # TODO compute middle points from these contour intersections
    y1DIFF = np.diff(y1)[::2]
    y1MIDDLE = np.add(y1[::2], np.divide(y1DIFF, 2)).astype("uint8")
    y2DIFF = np.diff(y2)[::2]
    y2MIDDLE = np.add(y2[::2], np.divide(y2DIFF, 2)).astype("uint8")
    # TODO extrapolate line to find k1-3
    k1 = findKPoints(largestContourIMG, y1MIDDLE[0], x1, y2MIDDLE[0], x2)
    k2 = findKPoints(largestContourIMG, y1MIDDLE[1], x1, y2MIDDLE[1], x2)
    k3 = findKPoints(largestContourIMG, y1MIDDLE[2], x1, y2MIDDLE[2], x2)

    # TODO calculate Rotation matrix from coordinate system spanned by k1-3
    rotMatrix = getCoordinateTransform(k1, k2, k3)
    # TODO rotate the image around new origin
    rotatedIMG = cv.warpAffine(img, rotMatrix, (img.shape[1], img.shape[0]))
    return rotatedIMG
