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
    def __init__(self, metadata, data_res, resolution, stride=8, exclude_chroms=['All', 'chrY','chrX', 'Y', 'X', 'chrM', 'M'], reference = 'mm9'):
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
        self.reference, self.data_res, self.resolution, self.split_res,  self.pixel_size = reference, data_res, resolution, int(resolution/stride), int(resolution/data_res)
        self.metadata = {'filename': metadata[0], 'replicate': metadata[1], 'norm': metadata[2], 'type_of_bin': metadata[3], 'class_id': metadata[4], 'chromosomes': OrderedDict()}
        self.positions = []
        self.exclude_chroms = exclude_chroms
        self.data = []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
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


class HiCDatasetDec(HiCDataset):
    """Hi-C dataset loader"""
    def __init__(self, *args, **kwargs):
        super(HiCDatasetDec, self).__init__(*args, **kwargs)
        straw_file = straw.straw(self.metadata['filename'])
        chromosomes = list(straw_file.chromDotSizes.data.keys() - self.exclude_chroms) 
        for chromosome in chromosomes: self.get_chromosome(straw_file,chromosome)
        self.data, self.metadata, self.positions = tuple(self.data), frozendict(self.metadata), tuple(self.positions)
    
    def add_chromosome(self, chromosome):
        if (chromosome in self.metadata['chromosomes'].keys()) |  (chromosome[3:] in self.metadata['chromosomes'].keys()): return print('chromosome already loaded')
        self.data, self.positions = list(self.data), list(self.positions)
        straw_file = straw.straw(self.metadata['filename'])
        self.get_chromosome(straw_file,chromosome)
        self.data, self.positions = tuple(self.data), tuple(self.positions)
    
    def get_chromosome(self, straw_file, chromosome):  
        straw_matrix = straw_file.getNormalizedMatrix(chromosome, chromosome, self.metadata['norm'], self.metadata['type_of_bin'], self.data_res)
        _, first, last = straw_file.chromDotSizes.figureOutEndpoints(chromosome)
        initial = len(self.positions)
        if 'chr' in chromosome: chromosome = chromosome[3:]
        self.metadata['chromosomes'][chromosome] = []
        for start_pos in range(first, last, self.split_res): self.make_matrix(straw_matrix,  start_pos, start_pos+self.resolution-self.
        data_res, chromosome) 
        self.metadata['chromosomes'][chromosome]= (initial, len(self.positions))
    
    def make_matrix(self, straw_matrix, start_pos, end_pos, chromosome):
        xpos, ypos, vals = straw_matrix.getDataFromGenomeRegion(start_pos, end_pos, start_pos, end_pos)
        if len(set(xpos))<self.pixel_size*0.8: return None
        xpos, ypos = np.array(xpos)-start_pos/straw_matrix.binsize, np.array(ypos)-start_pos/straw_matrix.binsize
        image_scp = csr_matrix( (vals, (xpos, ypos) ), shape=(self.pixel_size,self.pixel_size) ).toarray()
        image_scp[np.isnan(image_scp)] = 0
        image_scp = (image_scp+np.transpose(image_scp)-np.diag(np.diag(image_scp)))/np.nanmax(image_scp)
        image_scp = np.expand_dims(image_scp, axis=0)
        image_scp = torch.as_tensor(image_scp, dtype=torch.float)
        self.data.append((image_scp, self.metadata['class_id']))
        self.positions.append(start_pos)

class GroupedHiCDataset(HiCDataset):
    """Grouping multiple Hi-C datasets together"""
    def __init__(self, reference = 'mm9', resolution=880000, data_res=10000):
        self.reference, self.resolution, self.data_res = reference, resolution, data_res
        self.data,  self.metadata, self.starts, self.files = tuple(), [], [], set()   

    def add_data(self, dataset):
        if self.reference != dataset.reference: return print("incorrect reference")
        if self.resolution != dataset.resolution: return print("incorrect resolution")
        if self.data_res != dataset.data_res: return print("data resolutions do not match")
        if (dataset.metadata['filename'], dataset.metadata['norm']) in self.files: return print('file already in dataset with same normationsation')
        self.data = self.data + dataset.data
        self.metadata.append(dataset.metadata)
        self.starts.append(len(self.data))
        self.files.add( (dataset.metadata['filename'], dataset.metadata['norm']) )


class SiameseHiCDataset(HiCDataset):
    """Paired Hi-C datasets by genomic location."""
    def __init__(self, list_of_HiCDatasets, sims=(0,1), reference = ['mm9',{'1':197195432,'2':181748087,'3':159599783,'4':155630120,'5':152537259,'6':149517037,'7':152524553,'8':131738871,'9':124076172,'10':129993255,'11':121843856,'12':121257530,'13':120284312, '14':125194864,'15':103494974, '16':98319150,'17':95272651,'18':90772031,'19':61342430}], resolution=880000, stride=1, data_res=10000):
        self.sims,  self.resolution, self.data_res, self.split_res = sims, resolution, data_res, int(resolution/stride)
        self.reference, self.chromsizes = reference
        self.data =[]
        self.positions =[] 
        self.chromosomes = OrderedDict()
        checks = self.check_input(list_of_HiCDatasets)
        if not checks: return None
        self.make_data(list_of_HiCDatasets)
        self.metadata = tuple([data.metadata for data in list_of_HiCDatasets])
        
    def __getitem__(self, idx):
        data1, data2 = self.data[idx]
        similarity = (self.sims[0] if data1[1] == data2[1] else self.sims[1])
        return data1[0], data2[0], similarity

    def check_input(self, list_of_HiCDatasets):
        filenames_norm = set()
        for data in list_of_HiCDatasets:
            if not isinstance(data, HiCDataset): 
                print("List of HiCDatasets need to be a list containing only HiCDataset objects.") 
                return False
            if (data.metadata['filename'], data.metadata['norm']) in filenames_norm:  
                print("file has been passed twice with the same normalisation")
                return False
            filenames_norm.add((data.metadata['filename'], data.metadata['norm']))
        return True 

    def make_data(self, list_of_HiCDatasets):
        datasets = len(list_of_HiCDatasets)
        for chrom in self.chromsizes.keys():
            start_index = len(self.positions)
            starts = [list_of_HiCDatasets[i].metadata['chromosomes'].setdefault(chrom, (0,0))[0] for i in range(0, datasets)]
            ends = [list_of_HiCDatasets[i].metadata['chromosomes'].setdefault(chrom, (0,0))[1] for i in range(0, datasets)]
            positions = [list(list_of_HiCDatasets[i].positions[starts[i]:ends[i]]) for i in range(0, datasets)]
            for pos in range(0, self.chromsizes[chrom], 880000)[::-1]: 
                curr_data = []
                for i in range(0,datasets): 
                    if positions[i][-1:]==[pos]:
                         self.positions.append(pos)
                         curr_data.append((list_of_HiCDatasets[i][starts[i]+len(positions[i])-1][0],list_of_HiCDatasets[i][starts[i]+len(positions[i])-1][1], i) )
                         positions[i].pop()
                self.data.extend([(curr_data[k], curr_data[j]) for k in range(0,len(curr_data)) for j in range(k+1,len(curr_data))])                    
            self.chromosomes[chrom] =(start_index,len(self.positions)) 


