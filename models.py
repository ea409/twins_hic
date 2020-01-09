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

#try GAN - or mutual information type situation
#learn f(x) such that x is the wt and f(x) is the CTCFKO - then i have to clean and label in a different way.
#or x is the CTCFKO and f(x) is the DKO

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            
            nn.Conv2d(1, 96, kernel_size=11,stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

        )
        
        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128,2))

    def forward_once(self, x):
        # Forward pass 
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2


