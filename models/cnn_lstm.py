import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.basic_model import BasicModel

class conv_module(nn.Module):
    def __init__(self):
        super(conv_module, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(8)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(8)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x


class lstm_module(nn.Module):
    def __init__(self):
        super(lstm_module, self).__init__()
        self.lstm = nn.LSTM(input_size=8*4*4+9, hidden_size=96, num_layers=1)

    def forward(self, x):

        # the input for lstm is, by default, (L, N, H_in)
        # where L is length of sequence, N is batch size, and H_in is image shape
        x = x.permute(1, 0, 2)

        # output, (h_n, c_n) = self.lstm(x)
        # only use the last hidden state
        _, (h_n, _) = self.lstm(x)
        return h_n


class mlp_module(nn.Module):

    def __init__(self):
        super(mlp_module, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(96, 8)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CNN_LSTM_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_LSTM_MLP, self).__init__(args)
        self.conv = conv_module()
        self.lstm = lstm_module()
        self.mlp = mlp_module()
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.tags = self.build_tags()
        # self.register_buffer("tags", torch.tensor(self.build_tags(), dtype=torch.float))
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)

    def build_tags(self):
        tags = np.zeros((16, 9))
        tags[:8, :8] = np.eye(8)
        tags[8:, 8] = 1
        tags = torch.tensor(tags, dtype = torch.float)
        tags = tags.unsqueeze(0).expand(self.batch_size, -1, -1)
        tags = tags.cuda()
        return tags

    def forward(self, x):
        x = x.view(-1, 1, self.img_size, self.img_size) # entry-wise encoding
        features = self.conv(x)
        features = features.flatten(start_dim = -3)
        features = features.view(self.batch_size, 16, -1)
        features = torch.cat([features, self.tags], dim=-1)
        h_n = self.lstm(features)
        h_n = h_n.squeeze() # the layer number (2 for bi-directional LSTM) is included in the first dimension
        pred = self.mlp(h_n)
        return pred

    