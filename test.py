import torch 
from torch.utils.data import Dataset, DataLoader, RandomSampler
import sys
import os
from scipy.sparse import csr_matrix
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import cnn_main
import torch.nn.functional as F

model = cnn_main.ConvNet(num_classes)
model.load_state_dict(torch.load('model.ckpt'))

# Test the model
 # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with model.eval():
    correct = 0
    total = 0
    for i, data in enumerate(dataloader):
        imgs, labels = data
        imgs = Variable(imgs.type(torch.FloatTensor))
        labels = np.asarray(labels)
        labels = torch.from_numpy(labels)
        labels = labels.to(device)
        outputs = model(imgs)
        total += len(labels)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model test images: {} %'.format(100 * correct / total))
