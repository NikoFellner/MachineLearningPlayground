# Implement the histogram equalization in this file
import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_histo(img):
    # erstellen des Histograms
    # Da ein Bild in Grautönen liegt ein Wertebereich zwischen 0 und 255, als Datatype int
    # mit der funktion np.unique und der Flag return_counts wird mir die Anzahl der jeweiligen Werte (falls vorhanden)
    # zurückgegeben
    # diese werden dann (falls bestimmte Intensitätswerte nicht im Bild vorhanden sind) in einem zero-vektor den
    # richtigen indizes zugeordnet
    valueOccurences = np.unique(img, return_counts=True)
    histo = np.zeros(256, dtype=int)
    histo[valueOccurences[0]] = valueOccurences[1]
    return histo

def create_cumulativeDistribution(histo):
    # 1. normieren des histograms
    # 2. iteriere über das histogram und addiere alle Werte des normierten Vektors schrittweise auf
    # 3. gib den kummulierten Vektor zurück
    normalizedHist = np.divide(histo, np.sum(histo))
    cumulativeDistribution = np.zeros(256)
    for n in range(255):
        cumulativeDistribution[n] = np.sum(normalizedHist[0:n+1])

    plt.figure(3)
    plt.subplot(1, 2, 1)
    plt.plot(normalizedHist)
    plt.title("normalized histogram")
    plt.subplot(1, 2, 2)
    plt.plot(cumulativeDistribution)
    plt.title("cumulative distribution")
    plt.savefig("CxIllustration.png")
    return cumulativeDistribution

def histogram_equalization(img, Cx):
    # 1. minimalwert aus der cumulative distribution der > 0 ist
    # 2. zuordnen der Intensitätswerte aus dem Originalbild zu den zugehörigen Werten aus der cumulativeDistribution
    # 3. Berechnen der neuen Intensitätswerte
    #       imgNew = ((Cx[img]-CxMin)/(1-CxMin))*255
    # 4. imgNew zurückgeben
    CxMin = np.amin(Cx[Cx>0])
    imgCx = Cx[img]
    #filteredImage = np.multiply(np.divide(np.subtract(imgCx, CxMin),(1-CxMin)),500)
    filteredImage = np.divide(np.multiply(np.subtract(imgCx, CxMin),1),(1-CxMin))
    #filteredImage = np.multiply(np.divide(np.add(imgCx, CxMin), (1 + CxMin)), 255)
    return filteredImage

img = cv2.imread('hello.png', cv2.IMREAD_GRAYSCALE)
histo = create_histo(img)
Cx = create_cumulativeDistribution(histo)
filteredImage = histogram_equalization(img, Cx)

plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(img, 'gray')
plt.title('Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(filteredImage, 'gray')
plt.title('Histogram Equalization')
plt.axis('off')
plt.savefig("OriginalVsFiltered.png")
plt.figure(2)
plt.imshow(filteredImage, 'gray')
plt.title('Histogram euqalized image')
plt.axis('off')
plt.savefig("kitty.png")
plt.show()
