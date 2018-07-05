import json
from time import sleep


def load_label2char():
    label2char = {}
    with open("etl_label2char.txt", "r") as ins:
        for line in ins:
            line = line.strip()
            if line.endswith("___"):
                line += "  "
            _from, _to = line.split("___")
            if len(_from) == 1:
                _from = " " + _from
            label2char[_from] = _to
    return label2char


label2char = load_label2char()
# print(label2char)

# # render single image
# with open("nichijou_ascii_strings_2/frame_14.json") as f:
#     labels = json.load(f)
#     render = []
#     for line in labels:
#         render.append([label2char[item] for item in line])

#     print('=================================================================')
#     for line in render:
#         print("".join(line))
#     print('=================================================================')

# render video
with open("nichijou_ascii_strings_x3.json") as f:
    all_labels = json.load(f)
    for labels in all_labels:
        render = []
        for line in labels:
            render.append([label2char[item] for item in line])

        # print('=================================================================')
        for line in render:
            print("".join(line))
        # print('=================================================================')
        sleep(0.05)  #

