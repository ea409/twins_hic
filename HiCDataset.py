import pickle
import straw
import numpy as np
from torch import as_tensor as as_torch_tensor, float as torch_float
from collections import OrderedDict
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from frozendict import frozendict
import cooler
from reference_dictionaries import reference_genomes

class HiCDataset(Dataset):
    """Hi-C dataset."""
    def __init__(self, metadata, data_res, resolution, stride=8, exclude_chroms=['chrY','chrX', 'Y', 'X', 'chrM', 'M'], reference = 'mm9'):
        """
        Args:
        metadata: A list consisting of
            filepath: string
            replicate name: string
            norm: (one of <NONE/VC/VC_SQRT/KR>)
            type_of_bin: (one of 'BP' or 'FRAG')
            class id: containing an integer specifying the biological condition of the Hi-C file.
        data_res: The resolution for the Hi-C to be called in base pairs.
        resolution: the size of the overall region to be considered.
        stride: (optional) gives the number of images which overlap.
        """
        self.reference, self.data_res, self.resolution, self.split_res,  self.pixel_size = reference, data_res, resolution, int(resolution/stride), int(resolution/data_res)
        self.metadata = {'filename': metadata[0], 'replicate': metadata[1], 'norm': metadata[2], 'type_of_bin': metadata[3], 'class_id': metadata[4], 'chromosomes': OrderedDict()}
        self.positions = []
        self.exclude_chroms = exclude_chroms +['All', 'ALL', 'all']
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
        if (len(set(xpos))<self.pixel_size*0.9) |  (np.sum(np.isnan(vals)) > 0.5*len(vals)) : return None
        xpos, ypos = np.array(xpos)-start_pos/straw_matrix.binsize, np.array(ypos)-start_pos/straw_matrix.binsize
        image_scp = csr_matrix( (vals, (xpos, ypos) ), shape=(self.pixel_size,self.pixel_size) ).toarray()
        image_scp[np.isnan(image_scp)] = 0
        image_scp = (image_scp+np.transpose(image_scp)-np.diag(np.diag(image_scp)))/np.nanmax(image_scp)
        image_scp = np.expand_dims(image_scp, axis=0)
        image_scp = as_torch_tensor(image_scp, dtype=torch_float)
        self.data.append((image_scp, self.metadata['class_id']))
        self.positions.append(start_pos)

class GroupedHiCDataset(HiCDataset):
    """Grouping multiple Hi-C datasets together"""
    def __init__(self, list_of_HiCDataset = None, resolution=2560000, data_res=10000):
        #self.reference = reference
        self.resolution, self.data_res =  resolution, data_res
        self.data,  self.metadata, self.starts, self.files = tuple(), [], [], set()
        if list_of_HiCDataset is not None:
            if not isinstance(list_of_HiCDataset, list): print("list of HiCDataset is not list type") #stop running
            for dataset in list_of_HiCDataset: self.add_data(dataset)

    def add_data(self, dataset):
        if not isinstance(dataset, HiCDataset): return print("file not HiCDataset")
        #if self.reference != dataset.reference: return print("incorrect reference")
        if self.resolution != dataset.resolution: return print("incorrect resolution")
        if self.data_res != dataset.data_res: return print("data resolutions do not match")
        self.data = self.data + dataset.data
        self.metadata.append(dataset.metadata)
        self.starts.append(len(self.data))

class SiameseHiCDataset(HiCDataset):
    """Paired Hi-C datasets by genomic location."""
    def __init__(self, list_of_HiCDatasets, sims=(0,1), reference = reference_genomes["mm9"], resolution=2560000, stride=16, data_res=10000):
        self.sims,  self.resolution, self.data_res, self.split_res = sims, resolution, data_res, int(resolution/stride)
        self.reference, self.chromsizes = reference
        self.data =[]
        self.positions =[]
        self.chromosomes = OrderedDict()
        checks = self.check_input(list_of_HiCDatasets)
        if not checks: return None
        self.make_data(list_of_HiCDatasets)
        self.metadata = tuple([data.metadata for data in list_of_HiCDatasets])

    def check_input(self, list_of_HiCDatasets):
        filenames_norm = set()
        if not isinstance(list_of_HiCDatasets, list): print("list of HiCdatasets is not list type")
        for data in list_of_HiCDatasets:
            if not isinstance(data, HiCDataset):
                print("List of HiCDatasets need to be a list containing only HiCDataset objects.")
                return False
            # if (data.metadata['filename'], data.metadata['norm']) in filenames_norm:
            #     print("file has been passed twice with the same normalisation") #maybe make this a warning instead of not doing it
            filenames_norm.add((data.metadata['filename'], data.metadata['norm']))
        return True

    def check_input_parameters(self, dataset): #where we check if the dataset is compatible with what we want to do
        pass

    def append_data(self, curr_data, pos):
        self.data.extend([(curr_data[k][0], curr_data[j][0], (self.sims[0] if curr_data[k][1] == curr_data[j][1] else self.sims[1]) ) for k in range(0,len(curr_data)) for j in range(k+1,len(curr_data))])
        self.positions.extend( [(pos, k, j) for k in range(0,len(curr_data)) for j in range(k+1,len(curr_data))])

    def make_data(self, list_of_HiCDatasets):
        datasets = len(list_of_HiCDatasets)
        for chrom in self.chromsizes.keys():
            start_index = len(self.positions)
            starts, positions = [], []
            for i in range(0, datasets):
                start, end = list_of_HiCDatasets[i].metadata['chromosomes'].setdefault(chrom, (0,0))
                starts.append(start)
                positions.append(list(list_of_HiCDatasets[i].positions[start:end]))
            for pos in range(0, self.chromsizes[chrom], self.split_res)[::-1]:
                curr_data = []
                for i in range(0,datasets):
                    if positions[i][-1:]!=[pos]: continue
                    curr_data.append(list_of_HiCDatasets[i][starts[i]+len(positions[i])-1] )
                    positions[i].pop()
                self.add_data(curr_data, pos)
            self.chromosomes[chrom] =(start_index,len(self.positions))
        self.data = tuple(self.data)

class HiCDatasetCool(HiCDataset):
    """Hi-C dataset loader"""
    def __init__(self, metadata, resolution, **kwargs):
        """ metadata: A list consisting of
            filepath: string
            replicate name: string
            norm: (one of <None/cool_norm>)
            class id: containing an integer specifying the biological condition of the Hi-C file."""
        cl_file= cooler.Cooler(metadata[0])
        metadata[2] = (metadata[2]=="cool_norm")
        metadata.insert(3, "NA")
        super(HiCDatasetCool, self).__init__(metadata, cl_file.binsize, resolution, reference = cl_file.info["genome-assembly"], **kwargs)
        chromosomes = list(set(cl_file.chromnames) - set(self.exclude_chroms))
        for chromosome in chromosomes: self.get_chromosome(cl_file, chromosome)
        self.data, self.metadata, self.positions = tuple(self.data), frozendict(self.metadata), tuple(self.positions)

    def add_chromosome(self, chromosome):
        if (chromosome in self.metadata['chromosomes'].keys()) |  (chromosome[3:] in self.metadata['chromosomes'].keys()): return print('chromosome already loaded')
        self.data, self.positions = list(self.data), list(self.positions)
        cl_file = cooler.Cooler(self.metadata['filename'])
        self.get_chromosome(cl_file, chromosome)
        self.data, self.positions = tuple(self.data), tuple(self.positions)

    def get_chromosome(self, cl_file, chromosome):
        stride = int(self.split_res/self.data_res)
        cl_matrix = cl_file.matrix(balance = self.metadata['norm'])
        first, last = cl_file.extent(chromosome)
        initial = len(self.positions)
        self.metadata['chromosomes'][chromosome] = []
        for start_pos in range(first, last, stride): self.make_matrix(cl_matrix,  start_pos, first)
        self.metadata['chromosomes'][chromosome]= (initial, len(self.positions))

    def make_matrix(self, cl_matrix, start_pos, first):
        image_scp = cl_matrix[start_pos:start_pos+self.pixel_size, start_pos:start_pos+self.pixel_size]
        if (sum(np.diagonal(np.isnan(image_scp)|(image_scp==0))) > self.pixel_size*0.9) : return None
        image_scp[np.isnan(image_scp)] = 0
        image_scp = image_scp/np.nanmax(image_scp)
        image_scp = np.expand_dims(image_scp, axis=0)
        image_scp = as_torch_tensor(image_scp, dtype=torch_float)
        self.data.append((image_scp, self.metadata['class_id']))
        self.positions.append( int(self.data_res*(start_pos-first)))


class metriclearnpaired_HiCDataset(SiameseHiCDataset):
    """Paired Hi-C datasets by genomic location."""
    def __init__(self, *args, **kwargs):
        self.labels =[]
        super(metriclearnpaired_HiCDataset, self).__init__( *args, **kwargs)

    def append_data(self, curr_data, pos):
        for k in range(0,len(curr_data)):
            for j in range(k+1,len(curr_data)):
                x1, x2 = curr_data[k][0].numpy(), curr_data[j][0].numpy()
                x1, x2 = x1.flatten(), x2.flatten()
                self.data.extend(  [ np.vstack((x1,x2))] )
                self.labels.extend( [ self.sims[0] if curr_data[k][1] == curr_data[j][1] else self.sims[1] ] )
                self.positions.extend( (pos, k, j) )

class SSIM_HiCDataset(SiameseHiCDataset):
    """Paired Hi-C datasets by genomic location."""
    def __init__(self, *args, **kwargs):
        self.labels =[]
        super(SSIM_HiCDataset, self).__init__( *args, **kwargs)

    def append_data(self, curr_data, pos):
        for k in range(0,len(curr_data)):
            for j in range(k+1,len(curr_data)):
                x1, x2 = curr_data[k][0][0].numpy(), curr_data[j][0][0].numpy()
                self.data.extend(  [  (x1, x2) ] )
                self.labels.extend( [ self.sims[0] if curr_data[k][1] == curr_data[j][1] else self.sims[1] ] )
                self.positions.extend( (pos, k, j) )
