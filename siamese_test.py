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
    predicted = F.pairwise_distance(output1,output2)>0.5
    predicted=predicted.type(torch.FloatTensor)
    correct += float(torch.sum(predicted ==labels))
    total += len(labels)