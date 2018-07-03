import numpy as np
import random
from helpers.read_etl1 import import_data


def data_split(valid_size, test_size):

    data, label, label_char = import_data(image_brightness=16)
    # all 63 x 64

    for i in range(len(data)):
        tmp = np.zeros((1, 64))
        data[i] = np.concatenate([data[i], tmp], axis=0)
        # now they're all 64 x 64

    combined = list(zip(data, label_char))
    random.shuffle(combined)
    data, label_char = zip(*combined)
    data, label_char = list(data), list(label_char)

    # images
    x = np.stack(data, axis=0)  # 130k x 64 x 64
    # labels
    label2id = dict()
    id2label = []
    y = []
    for i in range(len(label_char)):
        _char = str(label_char[i])
        if _char not in label2id:
            label2id[_char] = len(id2label)
            id2label.append(_char)
        y.append(label2id[_char])
    y = np.array(y)

    assert len(y) > valid_size + test_size
    test_x, test_y = x[-test_size:], y[-test_size:]
    valid_x, valid_y = x[-(valid_size + test_size): -test_size], y[-(valid_size + test_size): -test_size]
    train_x, train_y = x[: -(valid_size + test_size)], y[: -(valid_size + test_size)]

    with open("etl1_id2label.txt", "w") as text_file:
        text_file.write("\n".join(id2label))

    np.save("etl1_images_train.npy", train_x)
    np.save("etl1_labels_train.npy", train_y)

    np.save("etl1_images_valid.npy", valid_x)
    np.save("etl1_labels_valid.npy", valid_y)

    np.save("etl1_images_test.npy", test_x)
    np.save("etl1_labels_test.npy", test_y)
    print("saved etl1 data into files")
