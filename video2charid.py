import cv2
import torch
import json
from tqdm import tqdm
import numpy as np
from helpers.inference import load_id2label, load_model, predict
classifier_img_size = 20
etl = "6"
model_checkpoint_path = "saved_models/model" + etl + ".pt"
use_cuda = True
batch_size = 32


def binarize(img):
    _, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imgf


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


def get_all_charids():
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


get_all_charids()
