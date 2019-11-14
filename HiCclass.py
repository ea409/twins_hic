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
from torchvision import transforms

class HiCDataset(Dataset):
    """Hi-C dataset."""
    def __init__(self, root_dir, metadata, data_res, resolution, split_res, transform):
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
        self.transform = transform 
    def mutate_metadata(self):
        metadata=self.metadata
        metadata['classification']=1
        metadata.loc[metadata.file.str.contains('DKO'), 'classification']=2
        metadata.loc[metadata.file.str.contains('WT'), 'classification']=0
        return metadata
    def __len__(self):
        return self.metadata.end.iloc[-1]       
    def __getitem__(self, idx):
        idx=int(idx)
        data_res=self.data_res
        metobj=self.metadata.loc[((self.metadata.first_index<=idx) & (self.metadata.end>idx))]
        suffix = str(metobj.index.tolist()[0])
        minmet = int(idx-metobj.first_index)*self.sub_res+ int(metobj.start)
        maxmet = minmet +self.resolution
        image= pd.DataFrame()
        for i in range(0,self.split_res): 
            img_name = os.path.join(self.root_dir, suffix + '_'+str(int(idx+i)))
            image = image.append(pd.read_csv(img_name, names=list(['x','y','vals']), sep='\t'))
        image = image[(image.y > minmet) & (image.y < maxmet)]
        image.vals=image.vals/np.sum(image.vals*2)
        image.x =  (image.x - minmet)/data_res
        image.y =  (image.y - minmet)/data_res
        image_scp = csr_matrix( (image.vals, (image.x.map(int), image.y.map(int)) ), shape=(self.pixel_size,self.pixel_size) ).toarray()
        image_scp = np.log(image_scp+np.transpose(image_scp)+1)/np.max(np.log(image_scp+1))
        img = np.expand_dims(image_scp, axis=0)
        img =torch.Tensor(img)
        transform=self.transform
        sample = (transform(img), int(metobj.classification.tolist()[0]) )
        return sample

def get_meta_index(metadata, logicals, train=True):
   concatrange=range(0,0)
   start=0
   for i, met in metadata.iterrows():
      if train==True:
         if any(x in met.file for x in logicals): 
            concatrange=np.concatenate((concatrange,range(start, met.first_index)))
            start=met.end
         if i==len(metadata):
            concatrange=np.concatenate((concatrange,range(start, metadata.end.iloc[-1])))
      else: 
         if any(x in met.file for x in logicals):
            concatrange=np.concatenate((concatrange,range(met.first_index, met.end)))
   return concatrange
