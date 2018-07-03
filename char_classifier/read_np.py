import numpy as np
from matplotlib import pyplot as plt
import pdb


x = np.load("etl1_images.npy")
y = np.load("etl1_labels.npy")
id2label = []
with open("etl1_id2label.txt", "r") as ins:
    array = []
    for line in ins:
        line = line.strip()
        if line == 'b"\' "':
            line = "' "
        elif line.startswith("b'") and line.endswith("'"):
            line = line[2:-1]
        else:
            line = line[1:-1]
        id2label.append(line)

test_num = 90
idx = np.random.randint(len(y), size=(9, 10))

for i in range(test_num):
    row = i // 10
    col = i % 10
    plt.subplot(9, 10, i + 1)
    plt.imshow(x[idx[row][col]], cmap='gray')
    print(row, col, y[idx[row][col]], id2label[y[idx[row][col]]])

plt.show()
pdb.set_trace()

label_char_set = set(id2label)
print('=============================================')
print(label_char_set)
print('=============================================')
print(len(label_char_set))
