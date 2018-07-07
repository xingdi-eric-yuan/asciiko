import numpy as np
import sys
sys.path.append(sys.path[0] + "/..")

import torch
from helpers.model import FancyNeuralNetworks


def load_id2label(etl="1"):
    id2label = []
    with open("etl" + etl + "_id2label.txt", "r") as ins:
        for line in ins:
            line = line.strip()
            if line == 'b"\' "':
                line = "' "
            elif line == 'b"\'\'"':
                line = "''"
            elif line.startswith("b'") and line.endswith("'"):
                line = line[2:-1]
            else:
                line = line[1:-1]
            id2label.append(line)
    return id2label


def load_model(model_checkpoint_path, use_cuda=False):
    # Set the random seed manually for reproducibility.
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        if not use_cuda:
            print("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
        else:
            torch.cuda.manual_seed(42)
    else:
        use_cuda = False  # Disable CUDA.
    model = FancyNeuralNetworks(enable_cuda=use_cuda)
    if use_cuda:
        model.cuda()

    if use_cuda:
        model.load_state_dict(torch.load(model_checkpoint_path))
    else:
        model.load_state_dict(torch.load(model_checkpoint_path, map_location={'cuda:0': 'cpu'}))
    return model


def predict(model, batch_x, use_cuda, etl="1"):
    model.eval()
    batch_pred = model.forward(batch_x, etl=etl)
    batch_pred = batch_pred.cpu().data.numpy()
    batch_pred = np.argmax(batch_pred, -1)  # batch
    return batch_pred
