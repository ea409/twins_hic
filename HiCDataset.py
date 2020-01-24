import glob as glob 
import pandas as pd 
import numpy as np 
from torch import Tensor
from torch.utils.data import Dataset
import sys
import os
from scipy.sparse import csr_matrix
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import pickle 

class HiCType:
    def __init__(self,indx, metadata, root_dir,  data_res, resolution,  split_res, pixel_size, sub_res, transform, depth=None):
        idx=int(indx)
        metobj=metadata.loc[((metadata.first_index<=idx) & (metadata.end>idx))]
        suffix = str(metobj.index.tolist()[0])
        minmet = int(idx-metobj.first_index)*sub_res+ int(metobj.start)
        maxmet = minmet +resolution
        image= pd.DataFrame()
        for i in range(0,split_res): 
            img_name = os.path.join(root_dir, suffix + '_'+str(int(idx+i)))
            image = image.append(pd.read_csv(img_name, names=list(['x','y','vals']), sep='\t'))
        image = image[(image.y > minmet) & (image.y < maxmet)]
        image.vals=image.vals/np.sum(image.vals*2)
        image.x =  (image.x - minmet)/data_res
        image.y =  (image.y - minmet)/data_res
        image_scp = csr_matrix( (image.vals, (image.x.map(int), image.y.map(int)) ), shape=(pixel_size,pixel_size) ).toarray()
        image_scp = np.log(image_scp+np.transpose(image_scp)+1)/np.max(np.log(image_scp+1))
        img = np.expand_dims(image_scp, axis=0)
        img =Tensor(img)
        self.data  = (transform(img), int(metobj.classification.tolist()[0]) )
        if depth is None:
            self.depth = 1
        else:
            self.depth = np.single(np.minimum(1.8*np.log(1+ np.log(1+depth.depth[idx]/100000000)**0.5)+0.1,1))


class HiCDataset(Dataset):
    """Hi-C dataset."""
    def __init__(self, root_dir,data_res, resolution, split_res, transform, stride=2, metadatapath=None, specify_ind=None, logicals=None, depthpath=None):
        """
        Args:
            root_dir (string): Directory with all the images + * if it all images in dir.
            metadatapatth (string): Path where the result of split_files on a sample is saved. If path not specified then metadata =None
            spec index (range): the index covering the scope of the set 
            logicals (tuple): of the form (logicals_on, logicals_off) 
        """
        self.create_file_metadata(root_dir,metadatapath)
        self.metadata = ( root_dir, #root dir 
                          data_res, #data resolution
                          resolution, #resolution
                          split_res, #split res 
                          int(resolution/data_res), #pixel size 
                          int(resolution/split_res), #sub res 
                          transform) #transform 
        self.stride = stride
        self.get_data(depthpath, specify_ind, logicals)
        self.fix_first_end_index(self.stride)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].data, self.data[idx].depth

    def fix_first_end_index(self, stride):
        self.file_metadata  =  self.file_metadata.reset_index(drop=True)
        total=0
        for ind, met in self.file_metadata.iterrows():
            self.file_metadata.at[ind,'first_index'] =total
            total+= int((met.end - met.first_index -1)/stride)+1
            self.file_metadata.at[ind,'end']= total

    def create_file_metadata(self, root_dir, metadatapath):
        if metadatapath is None: 
            metadata = pd.read_csv(root_dir+"/metadata.csv")
        else:
            metadata = pd.read_csv(metadatapath)
        metadata['classification']=1
        metadata.loc[metadata.file.str.contains('DKO'), 'classification']=2
        metadata.loc[metadata.file.str.contains('WT'), 'classification']=0
        metadata['chromosome']=metadata.file.str.split('/').apply(lambda x: x[-1]).str.split('_').apply(lambda x: x[1])
        self.file_metadata = metadata

    def get_data(self, depthpath, specify_ind, logicals):
        if specify_ind is None: 
            if logicals is None: 
                length = self.file_metadata.end.iloc[-1] 
                specify_ind = range(0, length, self.stride)
            else: 
                specify_ind = self.get_meta_index( *logicals)
        HiCs = [] 
        if depthpath is None:
            depth=None
        else: 
            depth=pd.read_csv(depthpath, delimiter=' ')
        for i in specify_ind:
            HiCs.append(HiCType(i, self.file_metadata, *self.metadata, depth))
        self.data=tuple(HiCs)

    def add_data(self, hicdataset):
        if (self.metadata[0:-1] == hicdataset.metadata[0:-1]):
            self.data = self.data + hicdataset.data
            self.file_metadata=self.file_metadata.append(hicdataset.file_metadata)
            self.fix_first_end_index(1)
        else:
            print('datasets not compatible')        
    def save(self,filename):
        with open(filename, 'wb') as output: 
            output.write(pickle.dumps(self))
            output.close()

    def filter_by_Rad21(self, rad21_file_path, threshold=500):
        Rad21 = pd.read_csv(rad21_file_path, delimiter=' ')
        Rad21 = Rad21[Rad21.cnt > threshold].copy()
        indices = tuple()
        for chromosome in Rad21.chr.unique():
            chr_subset = Rad21[(Rad21.chr==chromosome)].copy()
            for i, met in self.file_metadata[self.file_metadata.file.str.contains(chromosome +'_')].iterrows():
                if i==0:
                    chr_subset.pos=(chr_subset.pos-met.start)/self.metadata[5]
                    chr_subset = chr_subset[chr_subset.pos % self.stride ==0]
                    chr_subset.pos = chr_subset.pos/self.stride 
                indices+=tuple(chr_subset.pos.astype('int')+met.first_index)
        return indices

    def get_meta_index(self,logicals_on, logicals_off):
        specify_ind=range(0,0)
        for index, met in self.file_metadata.iterrows():
            if(~any(x in met.file for x in logicals_off) & any(x in met.file for x in logicals_on)):
                specify_ind=np.concatenate((specify_ind,range(met.first_index, met.end, self.stride)))
            else:
                self.file_metadata.drop(index, inplace=True)
        return specify_ind

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file: 
            unpickled = pickle.Unpickler(file)
            loadobj = unpickled.load()
        return loadobj



class SiameseHiCDataset(Dataset):
    def __init__(self, HiCDataset):
        self.data = None 
        self.metadata, self.data = self.make_data(HiCDataset)
    def make_data(self, HiCDataset):
        #to do make HiC Siamese Dataset. 
        #order goes CTCF, DKO, WT then grouped, R1, R2
        data=[]
        for chrom in HiCDataset.file_metadata.chromosome.unique():
            subset = HiCDataset.file_metadata[HiCDataset.file_metadata.chromosome==chrom].copy()
            a=np.array(subset.sort_values(by='file').first_index)
            indices = [(a[i],a[j]) for i in range(0,6) for j in np.array(range(0,6))[:i]]
            for ind in range(0, np.min(subset.end-subset.first_index)):
                for i, j in indices:
                     data.append((HiCDataset[i+ind], HiCDataset[j+ind]))
        return len(indices),tuple(data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        #data1, depth1, data2, depth2, class1==class2
        if self.data[idx][0][0][1]==self.data[idx][1][0][1]:
            sim = 1
        else: 
            sim = -1
        return self.data[idx][0][0][0], self.data[idx][0][1], self.data[idx][1][0][0], self.data[idx][1][1], sim 

if __name__ == "__main__":
    #Hi-C params
    resolution, split_res, data_res = 880000, 8, 10000 #Old params: 440000, 4, 5000
    transform = transforms.Compose([transforms.ToPILImage(),  transforms.ToTensor()])
    #make train 
    data=HiCDataset("10kb_allreps",  data_res, resolution, split_res, transform, logicals=(['R1'], ['TR3','TR4','chr2','R2'] ), stride=2, depthpath="sequencing_depth.csv")
    data.save("HiCDataset_10kb_R1")
    data=HiCDataset("10kb_allreps",  data_res, resolution, split_res, transform, logicals=(['R2'], ['TR3','TR4','chr2','R1'] ), stride=2, depthpath="sequencing_depth.csv")
    data.save("HiCDataset_10kb_R2")
    data=HiCDataset("10kb_allreps",  data_res, resolution, split_res, transform, logicals=(['R1R2'], ['TR3','TR4','chr2'] ), stride=2, depthpath="sequencing_depth.csv")
    data.save("HiCDataset_10kb_R1R2")
    #make test
    data = HiCDataset("10kb_allreps",  data_res, resolution, split_res, transform, logicals=(['chr2'], ['WTR3','WTR4']), stride=2, depthpath="sequencing_depth.csv")
    data.save("HiCDataset_10kb_allreps_test")