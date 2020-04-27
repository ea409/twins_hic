import torch.nn as nn
import torch
import torch.nn.functional as F

class FCNet(nn.Module):
    def __init__(self, num_classes):
        super(FCNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(88*88, 1500),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1500, 300),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(300, 120),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(120, num_classes),
            )
    def forward(self, x):
        x = x.view(-1, 88*88)
        x = self.classifier(x)
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



class DeepConvNet(nn.Module):
    def __init__(self, num_classes):
        super(DeepConvNet, self).__init__()
        convnat = nn.Sequential(
            nn.Conv2d(20,20, 3),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(20),
        )
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 3),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(20),
            convnat,
            convnat,
            convnat,
        )
        self.classifier = nn.Sequential(
            nn.Linear(20*9, num_classes),
            nn.ReLU(True),
            nn.Dropout(),
            )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 20*9)
        x = self.classifier(x)
        return x

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

class SiameseNetAttentiveLayer(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    # 128@42*42
            nn.MaxPool2d(2),   # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(), # 128@18*18
            nn.MaxPool2d(2), # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),   # 256@6*6
        )
        self.linear = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out


class SiameseNet2(nn.Module):
    def __init__(self):
        super(SiameseNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 16, 5),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Sequential(
            nn.Linear(16*19*19, 120),
            nn.LeakyReLU(True),
            #nn.Dropout(),
            nn.Linear(120, 20),
            nn.LeakyReLU(True),
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


class SiameseNet3(nn.Module):
    def __init__(self):
        super(SiameseNet3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 16, 5),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Sequential(
            nn.Linear(16*19*19, 300),
            nn.LeakyReLU(True),
            #nn.Dropout(),
            nn.Linear(300, 60),
            nn.LeakyReLU(True),
            nn.Linear(60, 6),
            nn.LeakyReLU(True),
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

class SiameseNet4(nn.Module):
    def __init__(self):
        super(SiameseNet4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 16, 5),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Sequential(
            nn.Linear(16*19*19, 120),
            nn.LeakyReLU(True),
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

class SiameseNet5(nn.Module):
    def __init__(self):
        super(SiameseNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 16, 5),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Sequential(
            nn.Linear(16*19*19, 120),
            nn.LeakyReLU(True),
            nn.Linear(120, 10),
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

class FCSiamese(nn.Module):
    def __init__(self):
        super(FCSiamese, self).__init__()
        self.features = nn.Sequential(
        )
        self.linear = nn.Sequential(
            nn.Linear(88*88, 42*42),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(42*42, 19*19),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(19*19, 120),
            nn.LeakyReLU(True),
            nn.Linear(120, 20),
            nn.ReLU(True),
            )
        self.distance = nn.CosineSimilarity()
    def forward_one(self, x):
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        #out = self.distance(out1, out2)
        return out1, out2
