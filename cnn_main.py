import pandas as pd 
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
#import Additional_Samplers
import HiCclass
import models
import torch 


#Hi-C params.
resolution, split_res, data_res = 880000, 8, 10000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToPILImage(),  transforms.ToTensor()])

metadata= pd.read_csv("10kb_allreps/metadata.csv")
dataset=HiCclass.HiCDataset("10kb_allreps", metadata, data_res, resolution, split_res, transform=transform)

indices_train = HiCclass.get_meta_index(dataset, [''],['TR3','TR4','chr2'])
train_sampler = torch.utils.data.SubsetRandomSampler(indices_train) 
#OR: Additional_Samplers.WeightedSubsetSampler() with either sequencing depth or 
# R4/R3 included but WT singletons less likely in train. 

#CNN params.
batch_size, num_classes, learning_rate =17, 3, 0.2
no_of_batches= int(len(indices_train)/batch_size)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler = train_sampler)

# Convolutional neural network (two convolutional layers)
model=models.ConvNet(num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

#  Training
for epoch in range(20):
    running_loss=0.0
    if (epoch % 5):
        torch.save(model.state_dict(), 'model_10kb_trained_all.ckpt')
    for i, data in enumerate(dataloader):
        inputs, labels = data
        #imgs, labels = data[0].to(device), data[1].to(device)
        # zero gradients 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
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
torch.save(model.state_dict(), 'model_10kb_trained_all.ckpt')

