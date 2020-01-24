import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
#from torch_plus import additional_samplers
from HiCDataset import HiCType, HiCDataset, SiameseHiCDataset
import models
import torch 


#Hi-C params.
#resolution, split_res, data_res = 880000, 8, 10000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = HiCDataset.load("HiCDataset_10kb_R1")
data = HiCDataset.load("HiCDataset_10kb_R2")
dataset.add_data(data)
del data

Siamese = SiameseHiCDataset(dataset)

train_sampler = torch.utils.data.RandomSampler(dataset) 

#CNN params.
batch_size, num_classes, learning_rate =17, 3, 0.2
no_of_batches= np.floor(len(Siamese )/batch_size)
dataloader = DataLoader(Siamese, batch_size=batch_size, sampler = train_sampler)

# Convolutional neural network (two convolutional layers)
model=models.SiameseNet()

# Loss and optimizer
criterion = nn.CosineEmbeddingLoss(reduction='none')
optimizer = optim.Adam(model.parameters())

#  Training
for epoch in range(30):
    running_loss=0.0
    if (epoch % 5):
        torch.save(model.state_dict(), 'Siamese_FirstTry.ckpt')
    for i, data in enumerate(dataloader):
        input1, depth1, input2, depth2,  labels = data
        labels = labels.type(torch.FloatTensor)
        # zero gradients 
        optimizer.zero_grad()
        output1, output2 = model(input1, input2)
        #loss = torch.mean(criterion(outputs, labels))  #normal cross entropy loss 
        loss = torch.mean(torch.mul(torch.mul(depth1,depth2).unsqueeze(1),criterion(output1, output2, labels))) #depth adjusted loss 
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % no_of_batches == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'
            .format(epoch+1, i, running_loss/no_of_batches))


torch.save(model.state_dict(), 'Siamese_FirstTry.ckpt')
