import cv2
import torch
import numpy as np
from inference import load_id2label, load_model, predict
classifier_img_size = 20
model_checkpoint_path = 'saved_models/model1.pt'
use_cuda = False
batch_size = 32


def binarize(img):
    _, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imgf


img = cv2.imread('test_images/tsubasa_2.png', 0)
img = binarize(img)
height, width = img.shape[0], img.shape[1]
print(height, width)

h = height // classifier_img_size
if height % classifier_img_size > 0:
    h += 1
h *= classifier_img_size

w = width // classifier_img_size
if width % classifier_img_size > 0:
    w += 1
w *= classifier_img_size
w, h = w * 2, h * 2

height, width = h, w
print(height, width)
img_resize = cv2.resize(img, (width, height))
img_resize = img_resize.astype('uint8')
img_resize = binarize(img_resize)
# invert colors cuz in the test images, edges are in black
img_resize = 255 - img_resize
n_vert, n_hori = height // classifier_img_size, width // classifier_img_size

subimgs = []
for h in range(n_vert):
    for w in range(n_hori):
        _img = img_resize[h * classifier_img_size: (h + 1) * classifier_img_size, w * classifier_img_size: (w + 1) * classifier_img_size]
        subimgs.append(_img)

# heuristics: subimage without any value --> SPACE
non_zeros = []
non_zeros_idx = []
for i, _img in enumerate(subimgs):
    if np.sum(_img) == 0:
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

# load model
model = load_model(model_checkpoint_path, use_cuda)

y = []
number_batch = (x.size(0) + batch_size - 1) // batch_size
for b in range(number_batch):
    batch_x = x[b * batch_size: (b + 1) * batch_size]
    batch_y = predict(model, batch_x, use_cuda)
    y += batch_y.tolist()

id2label = load_id2label()
res = [['SP' for j in range(n_hori)] for i in range(n_vert)]
labels = [id2label[item] for item in y]
for _l, idx in zip(labels, non_zeros_idx):
    res[idx // n_hori][idx % n_hori] = _l

print(res)
