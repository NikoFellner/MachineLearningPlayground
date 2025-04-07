'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''
import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(Rx, Ry):
    '''
    calculate similarities of Ring features
    :param Rx: Ring features of Person X
    :param Ry: Ring features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    # DRxy = 1/features * SUM(Rxi-Ryi) for all i
    distance = np.divide(np.sum(np.abs(np.subtract(Rx, Ry))), len(Rx))
    return distance


def calculate_Theta_Distance(Thetax, Thetay):
    '''
    calculate similarities of Fan features
    :param Thetax: Fan features of Person X
    :param Thetay: Fan features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    # lxx = SUM([Thetaxi- 1/feature*SUM(Thetaxi)]²) for all i
    lxx = np.sum(np.square(np.subtract(Thetax,np.divide(np.sum(Thetax), len(Thetax)))))
    # lyy = SUM([Thetayi - 1 / feature * SUM(Thetayi)]²) for all i
    lyy = np.sum(np.square(np.subtract(Thetay, np.divide(np.sum(Thetay), len(Thetay)))))
    # lxy = SUM([Thetaxi- 1/feature*SUM(Thetaxi)]*[Thetayi - 1 / feature * SUM(Thetayi)]) for all i
    lxy = np.sum(np.multiply(np.subtract(Thetay, np.divide(np.sum(Thetay), len(Thetay))),np.subtract(Thetax,np.divide(np.sum(Thetax), len(Thetax)))))
    distance = (1-(lxy*lxy/(lxx*lyy)))*100
    return distance
