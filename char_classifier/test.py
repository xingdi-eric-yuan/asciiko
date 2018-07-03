import struct
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
import pdb
from read_etl1 import import_data

data, label, label_char = import_data(image_brightness=16)
print(type(data), type(label), type(label_char))

test_num = 90
idx = np.random.randint(138549, size=(9, 10))

for i in range(test_num):
    row = i // 10
    col = i % 10
    plt.subplot(9, 10, i + 1)
    plt.imshow(data[idx[row][col]], cmap='gray')
    print(row, col, label[idx[row][col]], label_char[idx[row][col]])

plt.show()
# pdb.set_trace()

label_char_set = set(label_char)
print('=============================================')
print(label_char_set)
print('=============================================')
print(len(label_char_set))
