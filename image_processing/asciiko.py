import cv2
import numpy as np
from matplotlib import pyplot as plt


def color2gray(img):
    res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return res


def binarize(img):
    _, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imgf


img = cv2.imread('test_images/worldcup.png', 1)
img = color2gray(img)
biimg = binarize(img)

# edges = cv2.Canny(biimg, 100, 200)

plt.subplot(131), plt.imshow(img, cmap = 'gray')
plt.title('Origin'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(cv2.Canny(img, 199, 200), cmap = 'gray')
plt.title('Edge 1'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(cv2.Canny(biimg, 199, 200), cmap = 'gray')
plt.title('Edge 2'), plt.xticks([]), plt.yticks([])

plt.show()