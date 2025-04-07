from CannyEdgeDetector import canny
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


im = Image.open('contrast.jpg')
im = ImageOps.grayscale(im)
im = np.array(im)
res = canny(im)
plt.imshow(res, 'gray')
plt.show()
ContrastIm = Image.fromarray(res.astype("uint8"))
ContrastIm.save("contrastEdges.jpg")