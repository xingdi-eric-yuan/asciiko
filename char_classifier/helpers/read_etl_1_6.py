import struct
import os
from PIL import Image, ImageEnhance
import numpy as np
import cv2
record_len = 2052

# Meta Data should have the pattern:
# [Path, rec_len, JIS X Notation Index, Picture Index]
ETL6C_META = (('/ETL6/ETL6/ETL6C_01'),
              ('/ETL6/ETL6/ETL6C_02'),
              ('/ETL6/ETL6/ETL6C_03'),
              ('/ETL6/ETL6/ETL6C_04'),
              ('/ETL6/ETL6/ETL6C_05'),
              ('/ETL6/ETL6/ETL6C_06'),
              ('/ETL6/ETL6/ETL6C_07'),
              ('/ETL6/ETL6/ETL6C_08'),
              ('/ETL6/ETL6/ETL6C_09'),
              ('/ETL6/ETL6/ETL6C_10'),
              ('/ETL6/ETL6/ETL6C_11'),
              ('/ETL6/ETL6/ETL6C_12'))

ETL1C_META = (('/ETL1/ETL1/ETL1C_01'),
              ('/ETL1/ETL1/ETL1C_02'),
              ('/ETL1/ETL1/ETL1C_03'),
              ('/ETL1/ETL1/ETL1C_04'),
              ('/ETL1/ETL1/ETL1C_05'),
              ('/ETL1/ETL1/ETL1C_06'),
              ('/ETL1/ETL1/ETL1C_07'),
              ('/ETL1/ETL1/ETL1C_08'),
              ('/ETL1/ETL1/ETL1C_09'),
              ('/ETL1/ETL1/ETL1C_10'),
              ('/ETL1/ETL1/ETL1C_11'),
              ('/ETL1/ETL1/ETL1C_12'),
              ('/ETL1/ETL1/ETL1C_13'))


def minimum_square_bounding_box(img):
    # img: h x w
    vertical = np.sum(img, 1)  # h
    horizontal = np.sum(img, 0)  # w
    # vertical
    # 000011111100
    v_start = len(vertical) - len(np.trim_zeros(vertical, trim='f'))  # 4
    v_end = len(np.trim_zeros(vertical, trim='b'))  # 10
    v_len = v_end - v_start

    h_start = len(horizontal) - len(np.trim_zeros(horizontal, trim='f'))  # 4
    h_end = len(np.trim_zeros(horizontal, trim='b'))  # 10
    h_len = h_end - h_start

    if v_len <= 0 or h_len <= 0:
        return img

    res = img[v_start: v_end, h_start: h_end]  # v_len x h_len

    pad = 16
    if v_len <= 32 and h_len <= 32:
        res = np.concatenate([np.zeros((pad, h_len)), res, np.zeros((pad, h_len))], 0)
        res = np.concatenate([np.zeros((v_len + pad * 2, pad)), res, np.zeros((v_len + pad * 2, pad))], 1)

    elif h_len > v_len:
        diff = h_len - v_len
        half = diff // 2
        if half >= 1:
            res = np.concatenate([np.zeros((half, h_len)), res, np.zeros((half, h_len))], 0)
    elif v_len > h_len:
        diff = v_len - h_len
        half = diff // 2
        if half >= 1:
            res = np.concatenate([np.zeros((v_len, half)), res, np.zeros((v_len, half))], 1)
    return res


def binarize(img):
    _, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imgf


def import_data(etl="1", image_brightness=16):
    META_DATA = ETL1C_META if etl == "1" else ETL6C_META
    factors, labels, label_chars = [], [], []
    filename = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
    for row in range(len(META_DATA)):
        part = META_DATA[row]

        f_part, l_part, l_char_part = import_dataset_part(path=filename + part,
                                                          image_brightness=image_brightness)
        factors += f_part
        labels += l_part
        label_chars += l_char_part
    print("Imported Dataset")
    return factors, labels, label_chars


def import_dataset_part(path, image_brightness=16, size=(64, 63)):
    # W, H        :   Width and Hight of pictures
    filename = path
    feature, labels, label_chars = [], [], []
    # Pixel ratio for the pictures
    counter = 0
    (W, H) = size
    with open(filename, 'rb') as f:
        while(True):
            try:
                f.seek(counter * record_len)
                s = f.read(record_len)
                try:
                    r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
                    iF = Image.frombytes('F', (W, H), r[18], 'bit', 4)
                    iP = iF.convert('L')
                    # Adjust image brightness.
                    # This class can be used to control the brightness of an image.
                    # An enhancement factor of 0.0 gives a black image.
                    # A factor of 1.0 gives the original image.
                    # ------------------------------------------
                    enhancer = ImageEnhance.Brightness(iP)
                    # -------------------------------------
                    # Factor 1.0 always returns a copy of the original image,
                    # lower factors mean less color (brightness, contrast, etc),
                    # and higher values more. There are no restrictions on this value.
                    # --------------------------
                    iE = enhancer.enhance(image_brightness)
                    # --------------------------
                    # binarize, or other opencv function
                    img = np.asarray(iE)
                    img = binarize(img)
                    img = minimum_square_bounding_box(img)
                    img = cv2.resize(img, (20, 20))
                    img = img.astype('uint8')
                    img = binarize(img)
                    labels += [r[3]]
                    feature += [img]
                    label_chars += [r[1]]
                    print("parsing data from: ", path, counter)
                except:
                    break
            except:
                break
            counter += 1

    return feature, labels, label_chars
