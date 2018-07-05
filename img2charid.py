import cv2
import json
import torch
from tqdm import tqdm
import numpy as np
from helpers.inference import load_id2label, load_model, predict
classifier_img_size = 20
etl = "6"
model_checkpoint_path = "saved_models/model" + etl + ".pt"
use_cuda = False
batch_size = 32


def binarize(img):
    _, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imgf


def get_edges(img):
    img = cv2.equalizeHist(img)
    img = cv2.Canny(img, 100, 200)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    return img


def single_image_to_charids(img, new_img_w, new_img_h, model, id2label):
    assert new_img_w % classifier_img_size == 0
    assert new_img_h % classifier_img_size == 0
    img = binarize(img)
    height, width = new_img_h, new_img_w
    # riseze
    img_resize = cv2.resize(img, (new_img_w, new_img_h))
    img_resize = img_resize.astype('uint8')
    img_resize = binarize(img_resize)
    n_vert, n_hori = height // classifier_img_size, width // classifier_img_size

    subimgs = []
    for h in range(n_vert):
        for w in range(n_hori):
            _img = img_resize[h * classifier_img_size: (h + 1) * classifier_img_size, w * classifier_img_size: (w + 1) * classifier_img_size]
            subimgs.append(_img)

    # heuristics: subimage without any value --> SPACE
    non_zeros = []
    non_zeros_idx = []
    non_ratio = 0.1
    for i, _img in enumerate(subimgs):
        _tmp = _img.astype('float32') / 255.0
        if np.sum(_tmp) <= _img.shape[0] * _img.shape[1] * non_ratio:
            continue
        non_zeros.append(_img)
        non_zeros_idx.append(i)

    x = np.stack(non_zeros, 0)  # non_zeros x 20 x 20
    # 0-255 --> 0.0-1.0
    x = x.astype('float32') / 255.0
    x = x.reshape((x.shape[0], 1) + x.shape[1:])
    if use_cuda:
        x = torch.autograd.Variable(torch.from_numpy(x).type(torch.FloatTensor).cuda())
    else:
        x = torch.autograd.Variable(torch.from_numpy(x).type(torch.FloatTensor))

    y = []
    number_batch = (x.size(0) + batch_size - 1) // batch_size
    for b in range(number_batch):
        batch_x = x[b * batch_size: (b + 1) * batch_size]
        batch_y = predict(model, batch_x, use_cuda, etl=etl)
        y += batch_y.tolist()

    res = [['SP' for j in range(n_hori)] for i in range(n_vert)]
    labels = [id2label[item] for item in y]
    for _l, idx in zip(labels, non_zeros_idx):
        res[idx // n_hori][idx % n_hori] = _l
    return res


def get_charids_from_bunch_of_edge_frames():
    # for nichijou_op.mp4
    # load model
    model = load_model(model_checkpoint_path, use_cuda)
    ori_width, ori_height = 1280, 720
    id2label = load_id2label(etl=etl)
    # 2156 frames
    frames = []
    for i in tqdm(range(2156)):
        img = cv2.imread("nichijou/frame" + str(i) + ".png", 0)
        # originally 1280 x 720
        tmp = single_image_to_charids(img, ori_width * 3, ori_height * 3, model, id2label)
        frames += [tmp]

    with open("nichijou_ascii_strings_x3.json", "w") as text_file:
        json.dump(frames, text_file)


def get_new_image_size(img):
    # because both
    # img.width % classifier_img_size and
    # img.height % classifier_img_size should be 0
    height, width = img.shape[0], img.shape[1]
    h = height // classifier_img_size
    if height % classifier_img_size > 0:
        h += 1
    h *= classifier_img_size

    w = width // classifier_img_size
    if width % classifier_img_size > 0:
        w += 1
    w *= classifier_img_size
    return w, h


def get_charids_from_raw_image():
    img = cv2.imread('test_images/lenna.png', 0)
    img = get_edges(img)

    model = load_model(model_checkpoint_path, use_cuda)
    id2label = load_id2label(etl=etl)
    new_img_w, new_img_h = get_new_image_size(img)
    # to + resolution, enable the following line
    new_img_w, new_img_h = new_img_w * 20, new_img_h * 20
    charids = single_image_to_charids(img, new_img_w, new_img_h, model, id2label)
    with open("tmp.json", "w") as text_file:
        json.dump(charids, text_file)


def get_charids_from_edge_image():
    img = cv2.imread('test_images/tsubasa_2.png', 0)
    img = binarize(img)
    # invert colors cuz in the test images, edges are black while background is white
    img = 255 - img
    model = load_model(model_checkpoint_path, use_cuda)
    id2label = load_id2label(etl=etl)
    new_img_w, new_img_h = get_new_image_size(img)
    # to + resolution, enable the following line
    new_img_w, new_img_h = new_img_w * 4, new_img_h * 4
    charids = single_image_to_charids(img, new_img_w, new_img_h, model, id2label)
    with open("tmp.json", "w") as text_file:
        json.dump(charids, text_file)


get_charids_from_raw_image()
# get_charids_from_edge_image()
# get_charids_from_bunch_of_edge_frames()
