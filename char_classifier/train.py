import time
import os
import numpy as np
import argparse
import yaml
from tqdm import tqdm
from os.path import join as pjoin
import sys
sys.path.append(sys.path[0] + "/..")

import torch
import torch.nn.functional as F
from helpers.model import FancyNeuralNetworks
from helpers.dataset import data_split
from helpers.generic import batch_generator, generator_queue
wait_time = 0.01  # in seconds


def load_data_from_np(split="train"):
    x = np.load("etl1_images_" + split + ".npy")
    y = np.load("etl1_labels_" + split + ".npy")
    id2label = []
    with open("etl1_id2label.txt", "r") as ins:
        for line in ins:
            line = line.strip()
            if line == 'b"\' "':
                line = "' "
            elif line.startswith("b'") and line.endswith("'"):
                line = line[2:-1]
            else:
                line = line[1:-1]
            id2label.append(line)
    return x, y, id2label


def eval(model, batch_generator, batch_size, data_size):

    model.eval()
    number_batch = (data_size + batch_size - 1) // batch_size
    data_queue, _ = generator_queue(batch_generator, max_q_size=20)
    correct = 0.0
    for i in range(number_batch):
        # qgen train one batch
        generator_output = None
        while True:
            if not data_queue.empty():
                generator_output = data_queue.get()
                break
            else:
                time.sleep(wait_time)
        batch_x, batch_y = generator_output
        batch_pred = model.forward(batch_x)
        batch_pred = batch_pred.cpu().data.numpy()
        batch_y = batch_y.cpu().data.numpy()
        batch_pred = np.argmax(batch_pred, -1)  # batch
        correct += np.sum((batch_pred == batch_y).astype('float32'))
    return correct / float(data_size)


def train(config):
    # Set the random seed manually for reproducibility.
    np.random.seed(config['general']['seed'])
    torch.manual_seed(config['general']['seed'])
    if torch.cuda.is_available():
        if not config['general']['use_cuda']:
            print("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
        else:
            torch.backends.cudnn.deterministic = config['general']['cuda_deterministic']
            torch.cuda.manual_seed(config['general']['seed'])
    else:
        config['general']['use_cuda'] = False  # Disable CUDA.

    if not os.path.exists("etl1_images_train.npy"):
        data_split(3000, 3000)  # valid and test size

    batch_size = config['training']['scheduling']['batch_size']
    x_train, y_train, id2label = load_data_from_np("train")
    x_valid, y_valid, _ = load_data_from_np("valid")

    train_batch_generator = batch_generator(x_train, y_train, batch_size=batch_size, enable_cuda=config['general']['use_cuda'])
    valid_batch_generator = batch_generator(x_valid, y_valid, batch_size=batch_size, enable_cuda=config['general']['use_cuda'])
    train_data_queue, _ = generator_queue(train_batch_generator, max_q_size=20)

    model = FancyNeuralNetworks(enable_cuda=config['general']['use_cuda'])
    if config['general']['use_cuda']:
        model.cuda()

    init_learning_rate = config['training']['optimizer']['learning_rate']
    learning_rate_decay_ratio = config['training']['optimizer']['learning_rate_decay_ratio']
    learning_rate_decay_lowerbound = config['training']['optimizer']['learning_rate_decay_lowerbound']
    learning_rate_decay_patience = config['training']['optimizer']['learning_rate_decay_patience']

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=init_learning_rate)

    best_valid_acc = -10000
    _patience = 0
    current_learning_rate = init_learning_rate
    number_batch = (x_train.shape[0] + batch_size - 1) // batch_size
    for epoch in range(config['training']['scheduling']['epoch']):

        model.train()
        sum_loss = 0.0
        with tqdm(total=number_batch, leave=True, ncols=160, ascii=True) as pbar:
            for i in range(number_batch):
                # qgen train one batch
                generator_output = None
                while True:
                    if not train_data_queue.empty():
                        generator_output = train_data_queue.get()
                        break
                    else:
                        time.sleep(wait_time)
                batch_x, batch_y = generator_output
                optimizer.zero_grad()
                model.zero_grad()
                batch_pred = model.forward(batch_x)
                loss = F.nll_loss(batch_pred, batch_y)
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm(model.parameters(), config['training']['optimizer']['clip_grad_norm'])
                optimizer.step()

                batch_loss = loss.cpu().data.numpy()
                sum_loss += batch_loss * batch_size
                pbar.set_description('epoch=%d, batch=%d, avg_loss=%.5f, batch_loss=%.5f, lr=%.8f' % (epoch, i, sum_loss / float(batch_size * (i + 1)), batch_loss, current_learning_rate))
                pbar.update(1)

        valid_acc = eval(model, valid_batch_generator, batch_size, x_valid.shape[0])
        print("epoch = %d, valid accuracy = %.4f" % (epoch, valid_acc))
        # save & reload checkpoint by best valid performance
        model_checkpoint_path = config['training']['scheduling']['model_checkpoint_path']
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), model_checkpoint_path)
            print("========= saved checkpoint =========")
            _patience = 0
        else:
            # learning rate decay
            _patience += 1
            if _patience >= learning_rate_decay_patience:
                if current_learning_rate > init_learning_rate * learning_rate_decay_lowerbound:
                    current_learning_rate = current_learning_rate * learning_rate_decay_ratio
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_learning_rate
                _patience = 0


if __name__ == '__main__':
    for _p in ['saved_models']:
        if not os.path.exists(_p):
            os.mkdir(_p)
    parser = argparse.ArgumentParser(description="train network.")
    parser.add_argument("-c", "--config_dir", default='config', help="the default config directory")
    args = parser.parse_args()

    # Read config from yaml file.
    config_file = pjoin(args.config_dir, 'config.yaml')
    with open(config_file) as reader:
        config = yaml.safe_load(reader)

    train(config=config)
