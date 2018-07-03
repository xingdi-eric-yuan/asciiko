import struct
import os
from PIL import Image, ImageEnhance
import traceback
import logging
import numpy as np
import cv2

# Meta Data should have the pattern:
# [Path, categories, sheets, rec_len, JIS X Notation Index, Picture Index]
ETL1C_META = (('/ETL1/ETL1/ETL1C_01', 8, 1445, 2052, 3, 18),
              ('/ETL1/ETL1/ETL1C_02', 8, 1445, 2052, 3, 18),
              ('/ETL1/ETL1/ETL1C_03', 8, 1445, 2052, 3, 18),
              ('/ETL1/ETL1/ETL1C_04', 8, 1445, 2052, 3, 18),
              ('/ETL1/ETL1/ETL1C_05', 8, 1445, 2052, 3, 18),
              ('/ETL1/ETL1/ETL1C_06', 8, 1445, 2052, 3, 18),
              ('/ETL1/ETL1/ETL1C_07', 8, 1411, 2052, 3, 18),
              ('/ETL1/ETL1/ETL1C_08', 8, 1411, 2052, 3, 18),
              ('/ETL1/ETL1/ETL1C_09', 8, 1411, 2052, 3, 18),
              ('/ETL1/ETL1/ETL1C_10', 8, 1411, 2052, 3, 18),
              ('/ETL1/ETL1/ETL1C_11', 8, 1411, 2052, 3, 18),
              ('/ETL1/ETL1/ETL1C_12', 8, 1411, 2052, 3, 18),
              ('/ETL1/ETL1/ETL1C_13', 3, 1411, 2052, 3, 18))


FORBIDDEN_KATAKANA_SHEETS = list(
    range(1191, 1197)) + list(range(1234, 1243)) + [2011, 2911]


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


def import_data(image_brightness=16):
    factors, labels, label_chars = [], [], []
    filename = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
    for row in range(len(ETL1C_META)):
        part = ETL1C_META[row]
        num_sheets = part[2]

        f_part, l_part, l_char_part = import_dataset_part(path=filename + part[0],
                                                          num_of_categories=part[
                                                              1],
                                                          sheets=num_sheets,
                                                          record_len=part[3],
                                                          database='ETL1C',
                                                          image_brightness=image_brightness)
        factors += f_part
        labels += l_part
        label_chars += l_char_part
    print("Imported Dataset")
    return factors, labels, label_chars


def parse_data(f, record_len, num_char, size, database, image_brightness):
    # ARGS:
    #   f           :   The file which we are reading from
    #   record_len  :   The records length in bytes
    #   num_char    :   Number of copies of each char to parse
    #   size        :   Tuple with Width and Hight of pictures
    #   database    :   Defines which database data is read from
    feature_part = []
    labels_char_part = []
    labels_part = []
    (W, H) = size
    for i in range(0, num_char - 11):
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

            if r[2] not in FORBIDDEN_KATAKANA_SHEETS:
                labels_part += [r[3]]
                feature_part += [img]
                labels_char_part += [r[1]]

        except Exception as e:
            print('Parse Exception at ' + str(i))
            print(record_len)
            print(s)
            logging.error(traceback.format_exc())

    return feature_part, labels_part, labels_char_part


def import_dataset_part(path, num_of_categories,
                        sheets, record_len, database,
                        image_brightness=16, size=(64, 63), lower_edge_of_categories=0):
    # ARGS:
    #   sheets      :   The number of writers
    #   record_len  :   The records length in bytes
    #   W, H        :   Width and Hight of pictures
    #   categories  :   Defined as 8 for ETL1C-01~ETL1C-12, and 3 in ETL1C-13
    # filename = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/dataset' + dataset + part
    filename = path
    categories = range(lower_edge_of_categories, num_of_categories)
    feature, labels, label_chars = [], [], []
    # Pixel ratio for the pictures
    with open(filename, 'rb') as f:
        # Reads in from file here:
        for character in categories:
            # Skipping forward for each character:
            # ARGS:
            #   sheets      : The number of writers
            #   record_len  : The records length in bytes
            # --------------------------------------------

            f.seek((character * sheets + 1) * record_len)

            feature_part, labels_part, labels_char_part = parse_data(f=f,
                                                                     record_len=record_len,
                                                                     num_char=sheets,
                                                                     size=size,
                                                                     database=database,
                                                                     image_brightness=image_brightness)
            feature += feature_part
            labels += labels_part
            label_chars += labels_char_part

    return feature, labels, label_chars
