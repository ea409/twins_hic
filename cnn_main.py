import pandas as pd 
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import HiCclass
import models
import torch 

resolution, split_res, data_res = 880000, 8, 10000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToPILImage(),  transforms.ToTensor()])


# dataset_highres=HiCclass.HiCDataset("fast_data_access/cleaned_data", metadata_highres, data_res, resolution, split_res, transform=transform)
# metadata_highres= pd.read_csv("fast_data_access/cleaned_data_metadata.csv")

# indices_highres_test= np.concatenate((range( 15612, 17229), range(46485, 48102), range(60245, 61862)))
# indices_highres_train = np.concatenate((range(0, 15611), range(17229, 46484), range(48101, 60244), range(61861, len(dataset_highres))))

metadata= pd.read_csv("10kb_allreps/metadata.csv")
dataset=HiCclass.HiCDataset("10kb_allreps", metadata, data_res, resolution, split_res, transform=transform)

#metadata_lowres= pd.read_csv("fast_data_access/cleaned_data_low_res_metadata.csv")
#dataset_lowres=HiCclass.HiCDataset("fast_data_access/cleaned_data_low_res", metadata_lowres, data_res, resolution, split_res, transform=transform)

#indices_lowres_test= np.concatenate((range(35099, 36716), range(45083, 46700), range(46700, 48317), range(83842,  85459), range(99929, 101546), range(107792, 109409)))
#indices_lowres_train = np.concatenate((range(0, 35099), range(36716, 45083), range(48317, 83842), range(85459,99929), range(101546,107792), range(109409,len(dataset_lowres))))

indices_train = HiCclass.get_meta_index(metadata, ['TR3','TR4','chr2'], train=True)
indices_test = HiCclass.get_meta_index(metadata, ['TR3','TR4','chr2'], train=True)

#test_sampler =  torch.utils.data.SubsetRandomSampler(indices_lowres_test)
train_sampler = torch.utils.data.SubsetRandomSampler(indices_train)

batch_size, num_classes, learning_rate =17, 3, 0.2
no_of_batches= int(len(indices_train)/batch_size)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler = train_sampler) #change to split test and train and add on thee other dataset. 

# Convolutional neural network (two convolutional layers)
model=models.ConvNet(num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

#  Training
for epoch in range(20):
    running_loss=0.0
    if (epoch % 5):
        torch.save(model.state_dict(), 'model_10kb.ckpt')
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
torch.save(model.state_dict(), 'model_10kb.ckpt')

