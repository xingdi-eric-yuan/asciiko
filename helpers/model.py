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
        # A 7-layer convnet

        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(128, 192, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(192, 256, kernel_size=3)
        self.fc1 = torch.nn.Linear(2304, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3_etl1 = torch.nn.Linear(1024, 96)
        self.fc3_etl6 = torch.nn.Linear(1024, 114)
        self.dropout1 = torch.nn.Dropout(p=0.25)
        self.dropout2 = torch.nn.Dropout(p=0.25)
        self.dropout3 = torch.nn.Dropout(p=0.25)
        self.dropout4 = torch.nn.Dropout(p=0.5)
        self.dropout5 = torch.nn.Dropout(p=0.5)

    def forward(self, x, etl="1"):
        # x: batch x 1 x 20 x 20
        # conv layers
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # batch x 64 x 9 x 9
        x = F.relu(self.conv2(x))  # batch x 128 x 7 x 7
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))  # batch x 192 x 5 x 5
        x = self.dropout2(x)
        x = F.relu(self.conv4(x))  # batch x 256 x 3 x 3
        x = self.dropout3(x)
        # fc layers
        x = x.view(x.size(0), -1)  # batch x 2304
        x = F.relu(self.fc1(x))  # batch x 1024
        x = self.dropout4(x)
        x = F.relu(self.fc2(x))  # batch x 1024
        x = self.dropout5(x)
        # output layers
        if etl == "1":
            x = self.fc3_etl1(x)  # batch x 96
        elif etl == "6":
            x = self.fc3_etl6(x)  # batch x 114
        return F.log_softmax(x, dim=1)  # batch x 96
