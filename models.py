import torch.nn as nn
import torch
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 16, 5)
        self.fc1 = nn.Linear(16*19*19, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        #self.norm = nn.Softmin(dim=1)
        self.features = nn.Sequential(self.conv1, self.pool, self.conv2, self.pool) 
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 16*19*19)
        #x = x.view(-1, 88 * 88)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.norm(x)
        return x

class dConvNet(nn.Module):
    def __init__(self, num_classes):
        super(dConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5,dilation=2), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 16, 5,dilation=2), 
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*19*19, 120),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(84, num_classes),
            )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16*19*19)
        x = self.classifier(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 16, 5), 
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*19*19, 120),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(84, num_classes),
            )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16*19*19)
        x = self.classifier(x)
        return x
#try GAN - or mutual information type situation
#learn f(x) such that x is the wt and f(x) is the CTCFKO - then i have to clean and label in a different way.
#or x is the CTCFKO and f(x) is the DKO