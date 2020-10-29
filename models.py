import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

#learn f(x) such that x is the wt and f(x) is the CTCFKO - then i have to clean and label in a different way.
#or x is the CTCFKO and f(x) is the DKO

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 16, 5),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Sequential(
            nn.Linear(16*19*19, 120),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(120, 20),
            nn.ReLU(True),
            )
        self.distance = nn.CosineSimilarity()
    def forward_one(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        #out = self.distance(out1, out2)
        return out1, out2

class SLeNet(nn.Module):
    def __init__(self, mask=False):
        super(SLeNet, self).__init__()
        if mask:
            mask = np.ones((256, 256), int)
            np.fill_diagonal(mask, 0)
            self.mask = torch.tensor([mask])
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5, 1),
            nn.MaxPool2d(2, stride=2),
        )
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(16*61*61, 120),
            nn.GELU(),
            nn.Linear(120, 83),
            nn.GELU(),
            )
        self.distance = nn.CosineSimilarity()
    def forward_one(self, x):
        if hasattr(self, "mask"): x=self.mask*x
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        #out = self.distance(out1, out2)
        return out1, out2

class SAlexNet(nn.Module):
    def __init__(self,mask=False):
        super(SAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),
            nn.GELU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.GELU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.GELU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.GELU(),
            nn.Linear(in_features=4096, out_features=83),
        )
        self.distance = nn.CosineSimilarity()
    def forward_one(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        #out = self.distance(out1, out2)
        return out1, out2


class SZFNet(nn.Module):
    def __init__(self,mask=False):
        super(SZFNet, self).__init__()
        self.channels = 1
        self.conv_net = self.get_conv_net()
        self.fc_net = self.get_fc_net()
    def get_conv_net(self):
        layers = []
        # in_channels = self.channels, out_channels = 96
        # kernel_size = 7x7, stride = 2
        layer = nn.Conv2d(
            self.channels, 96, kernel_size=7, stride=2, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.GELU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.LocalResponseNorm(5))
        # in_channels = 96, out_channels = 256
        # kernel_size = 5x5, stride = 2
        layer = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.GELU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.LocalResponseNorm(5))
        # in_channels = 256, out_channels = 384
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.GELU())
        # in_channels = 384, out_channels = 384
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.GELU())
        # in_channels = 384, out_channels = 256
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.GELU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        return nn.Sequential(*layers)
    def get_fc_net(self):
        layers = []
        # in_channels = 9216 -> output of self.conv_net
        # out_channels = 4096
        layer = nn.Linear(256*7*7, 4096)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.Dropout())
        # in_channels = 4096
        # out_channels = self.class_count
        layer = nn.Linear(4096, 83)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.Dropout())
        return nn.Sequential(*layers)
    def forward_one(self, x):
        y = self.conv_net(x)
        y = y.view(-1, 7*7*256)
        y = self.fc_net(y)
        return y
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        #out = self.distance(out1, out2)
        return out1, out2

