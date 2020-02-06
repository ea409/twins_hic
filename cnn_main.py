import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch_plus import additional_samplers, loss 
from HiCDataset import HiCType, HiCDataset
import models
import torch 


#Hi-C params.
#resolution, split_res, data_res = 880000, 8, 10000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = HiCDataset.load("HiCDataset_10kb_R1")
data = HiCDataset.load("HiCDataset_10kb_R2")
dataset.add_data(data)
data = HiCDataset.load("HiCDataset_10kb_R1R2")
dataset.add_data(data)
del data

train_sampler = torch.utils.data.RandomSampler(dataset) 
#To exclude without Rad21 binding 
#train_sampler = torch.utils.data.SubsetRandomSampler(data.filter_by_Rad21("Cumulative_Rad21.txt", threshold=500)) 

#additional_samplers.WeightedSubsetSampler() with either sequencing depth or 
# R4/R3 included but WT singletons less likely in train. 

#CNN params.
batch_size, num_classes, learning_rate =17, 3, 0.2
no_of_batches= np.floor(len(dataset)/batch_size)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler = train_sampler)

# Convolutional neural network (two convolutional layers)
model=models.DeepConvNet(num_classes)

# Loss and optimizer
criterion = loss.DepthAdjustedLoss(nn.CrossEntropyLoss(reduction='none'))
optimizer = optim.Adam(model.parameters())

#  Training
for epoch in range(30):
    running_loss=0.0
    if (epoch % 5):
        torch.save(model.state_dict(), 'model_10kb_deep_cnn_with_depth_adjustment.ckpt')
    for i, data in enumerate(dataloader):
        data, depths = data
        inputs, labels =  data
        # zero gradients 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels, depths=None) # or depths=depths)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % 2000==0:
            _, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
            print(loss.item(), sum(predicted ==labels))
        if (i+1) % no_of_batches == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'
            .format(epoch+1, i, running_loss/no_of_batches))

# Save the model checkpoint
torch.save(model.state_dict(), 'model_10kb_deep_cnn_with_depth_adjustment.ckpt')

