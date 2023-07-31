import torch

import numpy as np
import torch.nn as nn
import h5py
import math
import time
import torch.nn.functional as F


class ML_Model(nn.Module):

    def __init__(self, in_dim, out_dim):
        """VGG

        ---
        in_dim = 128, out_dim = 2000 \\
        input_vec.shape = (1000, 128) \\
        output_vec.shape = (1000, 2000)

        ---
        Args:
            in_dim (int): query vector shape[-1]
            out_dim (int): num of leaf nodes
        """
        super(ML_Model, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = self.in_dim
        self.out_channel = self.in_dim
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, groups=32, bias=False),
            nn.ReLU6(inplace=True),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.ReLU6(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

        )
        self.hidden_dim = int((self.hidden_dim + 2 * 1 - 3) / 2) + 1  # after cnn  96*32 hidden_dim= 96

        self.hidden_dim = int(self.hidden_dim / 2)  # after pool 48*32  hidden_dim = 48


        self.out_channel = 32

        self.conv_layer = nn.Sequential(self.layer1)

        self.fc = nn.Sequential(nn.Linear(self.hidden_dim * self.out_channel, self.out_dim))
        # self.fc = nn.Sequential(
        #     nn.Linear(self.hidden_dim * self.out_channel, 10),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(10, self.out_dim)
        # )
        self.b = nn.BatchNorm1d(1)

    # cnn-fc1  cnn-fc2-fc3-attention      fc1*attention
    def forward(self, x):
        a = x.unsqueeze(1)
        # print(a.shape)
        a = self.conv_layer(a)  # [50, 64, 48]
        a = a.view(-1, self.hidden_dim * self.out_channel)


        # print("=============M1fc_input",self.hidden_dim * self.out_channel,self.out_dim)
        y = self.fc(a)

        return y


class ML_Model2(nn.Module):

    def __init__(self, in_dim, out_dim):
        """VGG

        ---
        in_dim = 128, out_dim = 2000 \\
        input_vec.shape = (1000, 128) \\
        output_vec.shape = (1000, 2000)

        ---
        Args:
            in_dim (int): query vector shape[-1]
            out_dim (int): num of leaf nodes
        """
        super(ML_Model2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = self.in_dim
        self.out_channel = self.in_dim

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(num_features=32), nn.ReLU(inplace=True),
            # nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm1d(num_features=32), nn.ReLU(inplace=True)
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 3, stride=2, padding=1), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.hidden_dim = int((self.hidden_dim + 2 * 1 - 3) / 2) + 1  # after cnn  96*32 hidden_dim= 96
        # self.hidden_dim = int(self.hidden_dim / 2)  # after pool 48*32  hidden_dim = 48、

        self.hidden_dim = int((self.hidden_dim + 2 * 1 - 3) / 2) + 1

        self.hidden_dim = int(self.hidden_dim / 2)


        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.hidden_dim = int((self.hidden_dim + 2 * 1 - 3) / 2) + 1
        self.hidden_dim = int(self.hidden_dim / 2)

        self.out_channel = 128

        self.conv_layer = nn.Sequential(self.layer1, self.layer2)
        # self.conv_layer = nn.Sequential(self.layer1,self.layer3)

        self.fc = nn.Sequential(nn.Linear(self.hidden_dim * self.out_channel, self.out_dim))

        self.b = nn.BatchNorm1d(1)

    # cnn-fc1  cnn-fc2-fc3-attention      fc1*attention
    def forward(self, x):
        a = x.unsqueeze(1)
        a = self.conv_layer(a)  # [50, 64, 48]
        # print(a.shape)
        a = a.view(-1, self.hidden_dim * self.out_channel)


        # print("=============M2fc_input", self.hidden_dim * self.out_channel,self.out_dim )
        y = self.fc(a)


        return y


class DSCNN_model(ML_Model):
    def __init__(self, in_dim, out_dim):
        ML_Model.__init__(self, in_dim, out_dim)

        self.divide_num = int(math.sqrt(self.out_dim))
        # self.divide_num = 5
        # self.divide_num = 10
        # self.divide_num = 20
        # self.divide_num = 50
        # self.divide_num = 150
        # self.divide_num = 200
        # print("divide_num = ", self.divide_num, "========================")
        self.M1 = ML_Model(self.in_dim, self.divide_num)
        # self.M2 = ML_Model2(self.divide_num + self.in_dim, self.divide_num)
        self.M2 = ML_Model2(self.divide_num + self.in_dim,self.divide_num )
        self.fc = nn.Sequential(
            nn.Linear(self.divide_num* self.divide_num, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, self.out_dim)
        )

    def forward(self, x):
        y1 = self.M1(x)  # [50, 34]
        x2 = torch.cat((x, y1), 1)  # [50, 225] hhhh

        y2 = self.M2(x2)  # [50, 34]

        y1 = y1.unsqueeze(-1)
        y2 = y2.unsqueeze(1)

        res = torch.bmm(y1, y2)  # [50, 34, 34]

        y = res.view(-1, res.shape[1] * res.shape[2])  # [50, 1156]

        y = self.fc(y)

        return y



