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

edge = cv2.Canny(biimg, 100, 200)

kernel = np.ones((2, 2), np.uint8)
dilation = cv2.dilate(edge, kernel, iterations=1)

plt.subplot(141), plt.imshow(img, cmap='gray')
plt.title('Origin'), plt.xticks([]), plt.yticks([])

plt.subplot(142), plt.imshow(biimg, cmap='gray')
plt.title('bi'), plt.xticks([]), plt.yticks([])

plt.subplot(143), plt.imshow(edge, cmap='gray')
plt.title('edge'), plt.xticks([]), plt.yticks([])

plt.subplot(144), plt.imshow(dilation, cmap='gray')
plt.title('edge dilation'), plt.xticks([]), plt.yticks([])


plt.show()
