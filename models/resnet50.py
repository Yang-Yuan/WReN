import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from models.basic_model import BasicModel



class mlp_module(nn.Module):
    def __init__(self):
        super(mlp_module, self).__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 8)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Resnet50_MLP(BasicModel):
    def __init__(self, args):
        super(Resnet50_MLP, self).__init__(args)
        resnet50 = models.resnet50(weights = None)
        resnet50.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet50_backbone = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
        self.mlp = mlp_module()
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)


    def forward(self, x):
        features = self.resnet50_backbone(x)

        # the last adaptiveAvg of ResNet uses kernel of size 1x1,
        # which introduces two trivial dimensions
        features = features.squeeze()

        score = self.mlp(features)
        return score

    