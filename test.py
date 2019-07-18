import torch 
from torch.utils.data import Dataset, DataLoader, RandomSampler
import sys
import os
from scipy.sparse import csr_matrix
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

model = torch.load('model.ckpt')

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for i, raw_imgs in enumerate(dataloader):
        imgs = Variable(raw_imgs['image'].type(torch.FloatTensor))
        str_labels = np.asarray(raw_imgs['type'])
        labels = np.asarray(labels)
        labels = torch.from_numpy(labels)
        labels = labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model test images: {} %'.format(100 * correct / total))
