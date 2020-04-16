import pandas as pd 
import numpy as np 
import pickle 
import straw 
import torch
from collections import OrderedDict 
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from frozendict import frozendict

class HiCDataset(Dataset):
    """Hi-C dataset."""
    def __init__(self, metadata, data_res, resolution, stride=8, exclude_choms=['All', 'chrY','chrX', 'Y', 'X', 'chrM', 'M']):
        """
        Args: 
        metadata: A list consisting of 
            filepath: string
            replicate name: string
            norm: (one of <NONE/VC/VC_SQRT/KR>)
            type_of_bin: (one of 'BP' or 'FRAG') 
            class id: containg an integer specifying the biological condition of the Hi-C file.
        data_res: The resolution for the Hi-C to be called in base pairs. 
        resolution: the size of the overall region to be considered. 
        stride: (optional) gives the number of images which overlap.
        """
        self.data_res, self.resolution, self.split_res,  self.pixel_size = data_res, resolution, int(resolution/stride), int(resolution/data_res)
        self.metadata = {'filename': metadata[0], 'replicate': metadata[1], 'norm': metadata[2], 'type_of_bin': metadata[3], 'class_id': metadata[4], 'chromosomes': OrderedDict()}
        straw_file = straw.straw(self.metadata['filename'])
        chromosomes = list(straw_file.chromDotSizes.data.keys() - exclude_choms) 
        self.data = []
        for i, chromosome in enumerate(chromosomes): self.get_chromosome(straw_file,chromosome, i)
        self.data, self.metadata = tuple(self.data), frozendict(self.metadata)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_chromosome(self, straw_file, chromosome, i):  
        straw_matrix = straw_file.getNormalizedMatrix(chromosome, chromosome, self.metadata['norm'], self.metadata['type_of_bin'], self.data_res)
        _, first, last = straw_file.chromDotSizes.figureOutEndpoints(chromosome)
        #if 'chr' in chromosome: chromosome = chromosome[3:]
        self.metadata['chromosomes'][chromosome] = []
        for start_pos in range(first, last, self.split_res): self.make_matrix(straw_matrix,  start_pos, start_pos+self.resolution-self.
        data_res, chromosome) 
        self.metadata['chromosomes'][chromosome]= tuple(self.metadata['chromosomes'][chromosome])

    def make_matrix(self, straw_matrix, start_pos, end_pos, chromosome):
        xpos, ypos, vals = straw_matrix.getDataFromGenomeRegion(start_pos, end_pos, start_pos, end_pos)
        if len(set(xpos))<self.pixel_size*0.8: return None
        xpos, ypos = np.array(xpos)-start_pos/straw_matrix.binsize, np.array(ypos)-start_pos/straw_matrix.binsize
        image_scp = csr_matrix( (vals, (xpos, ypos) ), shape=(self.pixel_size,self.pixel_size) ).toarray()
        image_scp = (image_scp+np.transpose(image_scp)-np.diag(np.diag(image_scp)))/np.max(image_scp)
        image_scp = np.expand_dims(image_scp, axis=0)
        image_scp = torch.as_tensor(image_scp, dtype=torch.float)
        self.data.append((image_scp, self.metadata['class_id']))
        self.metadata['chromosomes'][chromosome].append(start_pos)

    def save(self,filename):
        with open(filename, 'wb') as output: 
            output.write(pickle.dumps(self))
            output.close()

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file: 
            unpickled = pickle.Unpickler(file)
            loadobj = unpickled.load()
        return loadobj


class SiameseHiCDataset(Dataset):
    def __init__(self, HiCDataset, sims=(0,1)):
        self.data = []
        #self.metadata, self.data = self.make_data(HiCDataset)
        self.sims = sims

    def make_data(self, HiCDataset):
        pass

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        #data1, depth1, data2, depth2, class1==class2
        if self.data[idx][0][1]==self.data[idx][1][1]:
            sim = self.sims[0]
        else: 
            sim = self.sims[1]
        return self.data[idx][0][0], self.data[idx][1][0], sim 
