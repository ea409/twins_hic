import glob as glob 
import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader, RandomSampler
import sys
import os
from scipy.sparse import csr_matrix
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

resolution = 880000
split_res = 8
data_res = 5000
device='cpu'


class HiCDataset(Dataset):
    """Hi-C dataset."""
    def __init__(self, root_dir, metadata, data_res, resolution, split_res):
        """
        Args:
            root_dir (string): Directory with all the images + * if it all images in dir.
            metadata (pd.DataFrame): Result of split_files. 
                on a sample.
        """
        self.root_dir = root_dir
        self.metadata = metadata
        self.metadata = self.mutate_metadata()
        self.data_res = data_res
        self.resolution = resolution
        self.split_res = split_res 
        self.pixel_size = int(resolution/data_res)
        self.sub_res = int(resolution/split_res)
        
    def mutate_metadata(self):
        metadata=self.metadata
        metadata['classification']= 'CTCFKO'
        metadata.loc[metadata.file.str.contains('DKO'), 'classification']='DKO'
        metadata.loc[metadata.file.str.contains('WT'), 'classification']='WT'
        return metadata
    def __len__(self):
        return self.metadata.end.iloc[-1] 
    def __getitem__(self, idx):
        data_res=self.data_res
        metobj=self.metadata.loc[((self.metadata.first_index<=idx) & (self.metadata.end>idx))]
        suffix = str(metobj.index.tolist()[0])
        minmet = int(idx-metobj.first_index)*self.sub_res+ int(metobj.start)
        maxmet = minmet +self.resolution
        image= pd.DataFrame()
        for i in range(0,self.split_res): 
            img_name = os.path.join(self.root_dir, suffix + '_'+str(idx+i))
            image = image.append(pd.read_csv(img_name, names=list(['x','y','vals']), sep='\t'))
        image = image[(image.y > minmet) & (image.y < maxmet)]
        image.vals=image.vals/np.sum(image.vals)
        #fix all of this so that the output size is always the same 
        image.x =  (image.x - minmet)/data_res
        image.y =  (image.y - minmet)/data_res
        image_scp = csr_matrix( (image.vals, (image.x.map(int), image.y.map(int)) ), shape=(self.pixel_size,self.pixel_size) ).toarray()
        sample = {'image': np.expand_dims(image_scp, axis=0), 'type': str(metobj.classification.tolist()[0]) }
        return sample

metadata= pd.read_csv("test_code_metadata.csv")
dataset=HiCDataset("test_code", metadata, data_res, resolution, split_res)
sampler= RandomSampler(dataset,replacement=False)

dataloader = DataLoader(dataset,
                        batch_size=64,
                        sampler = sampler)

num_classes = 3
batch_size = 100
learning_rate = 0.001

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=64):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(61952, 30)
        self.fc2 = nn.Linear(30, num_classes)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#  Training
for epoch in range(30):
    for i, raw_imgs in enumerate(dataloader):
        imgs = Variable(raw_imgs['image'].type(torch.FloatTensor))
        str_labels = np.asarray(raw_imgs['type'])
        labels = []
        for label in str_labels:
            if label == 'WT':
                labels.append(0)
            elif label == 'CTCFKO':
                labels.append(1)
            else: 
                labels.append(2)
        labels = np.asarray(labels)
        labels = torch.from_numpy(labels)
        # print(labels)
        labels = labels.to(device)
        # Forward pass
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 1 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}' 
             .format(epoch+1, i, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for i, raw_imgs in enumerate(dataloader):
        imgs = Variable(raw_imgs['image'].type(torch.FloatTensor))
        # print(type(imgs), imgs.shape)
        str_labels = np.asarray(raw_imgs['type'])
        labels = []
        for label in str_labels:
            if label == 'WT':
                labels.append(0)
            elif label == 'CTCFKO':
                labels.append(1)
            else: 
                labels.append(2)
        labels = np.asarray(labels)
        labels = torch.from_numpy(labels)
        labels = labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        #print(predicted)
        #print(labels)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

