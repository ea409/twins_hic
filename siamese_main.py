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
import argparse

parser = argparse.ArgumentParser(description='Siamese network')
parser.add_argument('learning_rate',  type=float,
                    help='an integer for the accumulator')

args = parser.parse_args()

#Hi-C params.
#resolution, split_res, data_res = 880000, 8, 10000
cuda = torch.device("cuda:0")

dataset = HiCDataset.load("/vol/bitbucket/ealjibur/data/HiCDataset_10kb_R1")
data = HiCDataset.load("/vol/bitbucket/ealjibur/data/HiCDataset_10kb_R2")
dataset.add_data(data)
data = HiCDataset.load("/vol/bitbucket/ealjibur/data/HiCDataset_10kb_R1R2")
dataset.add_data(data)
del data

Siamese = SiameseHiCDataset(dataset)#,sims=(1,-1)

train_sampler = torch.utils.data.RandomSampler(Siamese) 

#CNN params.
batch_size, learning_rate = 17, args.learning_rate
no_of_batches= np.floor(len(Siamese )/batch_size)
dataloader = DataLoader(Siamese, batch_size=batch_size, sampler = train_sampler)

# Convolutional neural network (two convolutional layers)
model=models.SiameseNet().to(cuda)
model_save_path = 'outputs/Siamese_nodrop_LR'+str(learning_rate)+'.ckpt'
torch.save(model.state_dict(),model_save_path)

#validation 
dataset_validation =HiCDataset.load("/vol/bitbucket/ealjibur/data/HiCDataset_10kb_allreps_test_for_siamese")
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
        input1, _, input2, _,  labels = data
        input1, input2 = input1.to(cuda), input2.to(cuda)
        labels = labels.type(torch.FloatTensor).to(cuda)
        # zero gradients 
        optimizer.zero_grad()
        output1, output2 = model(input1, input2)
        loss = criterion(output1, output2, labels)  
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % no_of_batches == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'
            .format(epoch+1, i, running_loss/no_of_batches))

    for i, data in enumerate(dataloader_validation):
        input1, _, input2, _,  labels = data
        input1, input2 = input1.to(cuda), input2.to(cuda)
        labels = labels.type(torch.FloatTensor).to(cuda)
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
            
    
    torch.save(model.state_dict(), model_save_path)





