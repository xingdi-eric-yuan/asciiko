import numpy as np
import os
import cv2


def preprocessing(img):

    edges = cv2.Canny(img, 100, 200)
    kernel = np.ones((1, 1), np.uint8)
    # closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(edges, kernel, iterations=1)
    return dilation


def get_video_frames(path, folder_name):

    for _p in [folder_name]:
        if not os.path.exists(_p):
            os.mkdir(_p)

    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        image = preprocessing(image)
        cv2.imwrite(folder_name + "/frame%d.png" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
        # if count > 20:
        #     break


get_video_frames('test_images/nichijou_op.mp4', 'nichijou')
