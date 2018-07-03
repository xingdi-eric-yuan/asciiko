import numpy as np

import torch
import torch.nn.functional as F


class FancyNeuralNetworks(torch.nn.Module):
    model_name = 'fancynn'

    def __init__(self, enable_cuda=False):
        super(FancyNeuralNetworks, self).__init__()
        self._def_layers()
        self.print_parameters()

    def print_parameters(self):
        amount = 0
        for p in self.parameters():
            amount += np.prod(p.size())
        print("total number of parameters: %s" % (amount))
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        amount = 0
        for p in parameters:
            amount += np.prod(p.size())
        print("number of trainable parameters: %s" % (amount))

    def _def_layers(self):
        # M7-2 described in
        # https://github.com/charlietsai/japanese-handwriting-nn/blob/master/writeup.pdf

        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(128, 192, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(192, 256, kernel_size=3)
        self.fc1 = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 96)
        self.dropout1 = torch.nn.Dropout(p=0.25)
        self.dropout2 = torch.nn.Dropout(p=0.25)
        self.dropout3 = torch.nn.Dropout(p=0.25)
        self.dropout4 = torch.nn.Dropout(p=0.5)
        self.dropout5 = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        # x: batch x 1 x 64 x 64
        # conv layers
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # batch x 64 x 31 x 31
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # batch x 128 x 14 x 14
        x = self.dropout1(x)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))  # batch x 192 x 6, 6
        x = self.dropout2(x)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))  # batch x 256 x 2 x 2
        x = self.dropout3(x)
        # fc layers
        x = x.view(x.size(0), -1)  # batch x 4096
        x = F.relu(self.fc1(x))  # batch x 1024
        x = self.dropout4(x)
        x = F.relu(self.fc2(x))  # batch x 1024
        x = self.dropout5(x)
        # output layers
        x = self.fc3(x)  # batch x 96
        return F.log_softmax(x, dim=1)  # batch x 96
