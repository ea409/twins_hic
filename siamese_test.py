import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
#from torch_plus import additional_samplers
from HiCDataset import HiCType, HiCDataset, SiameseHiCDataset
import models
import torch 
from torch_plus.loss import ContrastiveLoss
import matplotlib.pyplot as plt 

dataset=HiCDataset.load("HiCDataset_10kb_allreps_test_for_siamese")
Siamese = SiameseHiCDataset(dataset)
test_sampler = SequentialSampler(Siamese)
dataloader = DataLoader(Siamese, batch_size=100, sampler = test_sampler)  

model=models.SiameseNet()
model.load_state_dict(torch.load('Siamese_SecondTry.ckpt'))
model.eval()

# Test the model
correct = 0
total = 0
for i, data in enumerate(dataloader):
    input1, depth1, input2, depth2,  labels = data
    labels = labels.type(torch.FloatTensor)
    # zero gradients 
    output1, output2 = model(input1, input2)
    predicted = F.pairwise_distance(output1,output2)>1.3
    predicted=predicted.type(torch.FloatTensor)
    correct += float(torch.sum(predicted ==labels))
    total += len(labels)



dataset=HiCDataset.load("HiCDataset_10kb_allreps_test_for_siamese")
data = HiCDataset.load("HiCDataset_10kb_R1")
dataset.add_data(data)
data = HiCDataset.load("HiCDataset_10kb_R2")
dataset.add_data(data)
del data
Siamese = SiameseHiCDataset(dataset)
test_sampler = SequentialSampler(Siamese)

dataloader = DataLoader(Siamese, batch_size=100, sampler = test_sampler)  


distances = np.array([])
indices = np.array([])
for i, data in enumerate(dataloader):
    input1, depth1, input2, depth2,  labels = data
    labels = labels.type(torch.FloatTensor)
    indices = np.concatenate((indices, labels.numpy()))
    # zero gradients 
    output1, output2 = model(input1, input2)
    predicted = F.pairwise_distance(output1,output2)
    distances = np.concatenate((distances, predicted.detach().numpy()))


plt.hist(distances[indices==0],100, density=True)
plt.hist(distances[indices==1],100, density=True)
plt.axvline(1.3)
plt.show()