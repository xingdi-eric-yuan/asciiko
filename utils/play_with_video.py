import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.animation as animation


def color2gray(img):
    res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return res


def canny_edge(img):
    edges = cv2.Canny(img, 100, 200)
    kernel = np.ones((1, 1), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)
    return dilation


def updatefig(*args):
    ret, frame = cap.read()
    # img = color2gray(frame)
    img = frame
    img = canny_edge(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    im.set_array(img)
    return im


cap = cv2.VideoCapture('test_images/nichijou_op.mp4')
while(cap.isOpened()):

    ret, frame = cap.read()
    # img = color2gray(frame)
    img = frame
    img = canny_edge(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # The following is the replacement for cv2.imshow():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(img, animated=True)
    ani = animation.FuncAnimation(fig, updatefig, interval=10)
    plt.show()
