import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
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


#Hi-C params.
#resolution, split_res, data_res = 880000, 8, 10000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = HiCDataset.load("HiCDataset_10kb_R1")
data = HiCDataset.load("HiCDataset_10kb_R2")
dataset.add_data(data)
del data

Siamese = SiameseHiCDataset(dataset)#,sims=(1,-1)

train_sampler = torch.utils.data.RandomSampler(Siamese) 

#CNN params.
batch_size, num_classes, learning_rate =17, 3, 0.1 #0.05 #0.2
no_of_batches= np.floor(len(Siamese )/batch_size)
dataloader = DataLoader(Siamese, batch_size=batch_size, sampler = train_sampler)

# Convolutional neural network (two convolutional layers)
model=models.SiameseNet()
torch.save(model.state_dict(), 'Siamese_nodrop_LR0_1.ckpt')

#validation 
dataset_validation =HiCDataset.load("HiCDataset_10kb_allreps_test_for_siamese")
Siamese_validation = SiameseHiCDataset(dataset_validation)
test_sampler = SequentialSampler(Siamese_validation)
batches_validation = np.ceil(len(dataset_validation)/100)
dataloader_validation = DataLoader(Siamese_validation, batch_size=100, sampler = test_sampler)  

# Loss and optimizer
criterion = ContrastiveLoss() #torch.nn.CosineEmbeddingLoss() #
optimizer = optim.Adagrad(model.parameters())

#  Training
for epoch in range(30):
    running_loss=0.0 
    running_validation_loss = 0.0
    for i, data in enumerate(dataloader):
        input1, depth1, input2, depth2,  labels = data
        labels = labels.type(torch.FloatTensor)
        # zero gradients 
        optimizer.zero_grad()
        output1, output2 = model(input1, input2)
        loss = criterion(output1, output2, labels)  
        #loss = torch.mean(torch.mul(torch.mul(depth1,depth2).unsqueeze(1),criterion(output1, output2, labels))) #depth adjusted loss 
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % no_of_batches == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'
            .format(epoch+1, i, running_loss/no_of_batches))

    for i, data in enumerate(dataloader_validation):
        input1, depth1, input2, depth2,  labels = data
        labels = labels.type(torch.FloatTensor)
        output1, output2 = model(input1, input2)
        loss = criterion(output1, output2, labels) 
        running_validation_loss += loss.item()

    print ('Epoch [{}/{}], Loss: {:.4f}'
            .format(epoch+1, i, running_validation_loss/batches_validation ))
    if (epoch>0):
        prev_validation_loss = min(prev_validation_loss,running_validation_loss)
        if (float(prev_validation_loss) +  0.1 < float(running_validation_loss)):
            break
    else: 
        prev_validation_loss =  running_validation_loss
            
    
    torch.save(model.state_dict(), 'Siamese_nodrop_LR0_1.ckpt')





