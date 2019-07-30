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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        metadata['classification']=1
        metadata.loc[metadata.file.str.contains('DKO'), 'classification']=2
        metadata.loc[metadata.file.str.contains('WT'), 'classification']=0
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
        image.vals=image.vals/np.sum(image.vals*2)
        image.x =  (image.x - minmet)/data_res
        image.y =  (image.y - minmet)/data_res
        image_scp = csr_matrix( (image.vals, (image.x.map(int), image.y.map(int)) ), shape=(self.pixel_size,self.pixel_size) ).toarray()
        image_scp = 100000*(image_scp+np.transpose(image_scp))
        img = np.expand_dims(image_scp, axis=0)
        sample = (torch.Tensor(img), int(metobj.classification.tolist()[0]) )
        return sample

metadata= pd.read_csv("cleaned_data_metadata.csv")
dataset=HiCDataset("cleaned_data", metadata, data_res, resolution, split_res)
sampler= RandomSampler(dataset,replacement=False)
batch_size=8

dataloader = DataLoader(dataset, batch_size=batch_size, sampler = sampler)

num_classes = 3
learning_rate = 0.2
no_of_batches= int(len(dataset)/batch_size)

# Convolutional neural network (two convolutional layers)
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 16, 5)
        self.fc1 = nn.Linear( 4 * 82 * 82, 120)
        #self.fc1 = nn.Linear( 176 * 176, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.norm = nn.Softmin(dim=1)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4 * 82 * 82)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.norm(x)
        return x

#model = ConvNet(num_classes).to(device)
model=Net(num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

#  Training
for epoch in range(20):
    running_loss=0.0
    for i, data in enumerate(dataloader):
        imgs, labels = data
        imgs = Variable(imgs.type(torch.FloatTensor))
        labels=labels.type(torch.LongTensor)
        #imgs, labels = data[0].to(device), data[1].to(device)
        # zero gradients 
        optimizer.zero_grad()
        # forward pass 
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        # backward pass 
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % no_of_batches == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'
            .format(epoch+1, i, running_loss/no_of_batches))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

