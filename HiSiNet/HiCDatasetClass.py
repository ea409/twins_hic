import pickle
import numpy as np
from torch import as_tensor as as_torch_tensor, float as torch_float
from collections import OrderedDict
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from frozendict import frozendict
from HiSiNet.reference_dictionaries import reference_genomes
from skimage.transform import resize
from skimage.morphology import convex_hull_image
from scipy import ndimage

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
    
    def get_genomic_positions(self, append=''):
        " returns chromosome, start, end "
        try:
            chromosome_metadata = self.chromosomes.items()
        except:
            chromosome_metadata = self.metadata['chromosomes'].items()
        chromosomes = np.concatenate( [ np.repeat(append+chromname, chromosome_range[1]-chromosome_range[0]) for 
                                    chromname, chromosome_range in chromosome_metadata])
        return {"Chromosome": chromosomes, "Start":  np.array(self.positions),"End": np.array(self.positions)+self.resolution}

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            unpickled = pickle.Unpickler(file)
            loadobj = unpickled.load()
        return loadobj


class HiCDatasetDec(HiCDataset):
    import hicstraw as straw
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
    def __init__(self, list_of_HiCDatasets):
        #self.reference = reference
        self.data,  self.metadata, self.starts, self.files = tuple(), [], [], set()
        if not isinstance(list_of_HiCDatasets, list): print("list of HiCDataset is not list type") #stop running
        self.resolution, self.data_res = list_of_HiCDatasets[0].resolution, list_of_HiCDatasets[0].data_res
        for dataset in list_of_HiCDatasets: self.add_data(dataset)

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
    def __init__(self, list_of_HiCDatasets, sims=(0,1), reference = reference_genomes["mm9"]):
        self.sims = sims
        self.reference, self.chromsizes = reference
        self.data =[]
        self.positions =[]
        self.labels = []
        self.chromosomes = OrderedDict()
        checks = self.check_input(list_of_HiCDatasets)
        if not checks: return None
        self.resolution, self.data_res, self.split_res = list_of_HiCDatasets[0].resolution, list_of_HiCDatasets[0].data_res, list_of_HiCDatasets[0].split_res
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
        self.positions.extend( [pos for k in range(0,len(curr_data)) for j in range(k+1,len(curr_data))])
        self.labels.extend( [( k, j) for k in range(0,len(curr_data)) for j in range(k+1,len(curr_data))])

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
                self.append_data(curr_data, pos)
            self.chromosomes[chrom] =(start_index,len(self.positions))
        self.data = tuple(self.data)

class HiCDatasetCool(HiCDataset):
    import cooler
    """Hi-C dataset loader"""
    def __init__(self, metadata, resolution, **kwargs):
        """ metadata: A list consisting of
            filepath: string
            replicate name: string
            norm: (one of <None/cool_norm/KR/VC/VC_SQRT>)
            class id: containing an integer specifying the biological condition of the Hi-C file."""
        cl_file= cooler.Cooler(metadata[0])
        metadata[2] = True if (metadata[2]=="cool_norm") else metadata[2] 
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
        

class PairOfDatasets(SiameseHiCDataset):
    from scipy import ndimage
    """Paired Hi-C datasets by genomic location to create feature map."""
    def __init__(self, list_of_HiCDatasets, model, **kwargs):
        self.model = model
        super(PairOfDatasets, self).__init__(list_of_HiCDatasets, **kwargs)
        self.pixel_size = int(self.resolution/self.data_res)
        self.paired_maps = {chromosome: self.make_maps(chromosome) for chromosome in self.chromosomes.keys()} 
        
    def append_data(self, curr_data, pos):
        self.data.extend([(self.model.features(curr_data[k][0].unsqueeze(0)).detach().numpy() - self.model.features(curr_data[j][0].unsqueeze(0)).detach().numpy())[0]  for k in range(0,len(curr_data)) for j in range(k+1,len(curr_data))])
        self.positions.extend( [pos for k in range(0,len(curr_data)) for j in range(k+1,len(curr_data))])
        self.labels.extend( [( k, j) for k in range(0,len(curr_data)) for j in range(k+1,len(curr_data))])
        
    def make_maps_base(self, chromosome, diagonal_off=4):
        nfilter = self.model.features[-2].out_channels
        chrom_index1, chrom_index2 = self.chromosomes[chromosome]
        if (chrom_index2==chrom_index1): return None
        dims = (nfilter, self.pixel_size , int((self.positions[chrom_index1])/self.data_res) + self.pixel_size)
        pair_maps = {}
        dataset_dims=(self.pixel_size ,self.pixel_size)

        for (map1,map2) in [(j,i) for j in range(0,len(self.metadata)) for i in range(j+1,len(self.metadata))]:
            pair_maps[(map1,map2)] = {}
            pair_maps[(map1,map2)]["rotated_shapes"] = np.zeros(dims)
            pair_maps[(map1,map2)]["norm"] = np.zeros(dims) 

        for ind in range(chrom_index2, chrom_index1, -1):
            true_ind=ind-1
            for curr_filt in range(0, nfilter):
                x = self.data[true_ind][curr_filt]
                x = (x+np.transpose(x))/2
                x = np.multiply(x,(np.tril(np.ones(x.shape[0]), -diagonal_off)+np.triu(np.ones(x.shape[1]), k=diagonal_off)))
                rotated = self.ndimage.rotate(x, angle=45, reshape=True)
                #if np.all(np.isnan(rotated )): return None 
                rotated = resize(rotated, dataset_dims)
                pair_maps[self.labels[true_ind]]["rotated_shapes"][curr_filt, :, int(self.positions[true_ind]/self.data_res):int((self.positions[true_ind]+self.resolution)/self.data_res)]+=rotated
                pair_maps[self.labels[true_ind]]["norm"][curr_filt, :,int(self.positions[true_ind]/self.data_res):int((self.positions[true_ind]+self.resolution)/self.data_res)]+=1
       
        all_maps= {pair:pair_maps[pair]["rotated_shapes"]/pair_maps[pair]["norm"] for pair in pair_maps.keys()}
        return all_maps
    
    def make_maps_grouped(self,all_maps):
        if all_maps is None: return None
        all_maps_grouped = {}
        maps_metadata = {}
        maps_metadata = { i:metadata["class_id"]  for i, metadata in enumerate(self.metadata)}
        all_maps_grouped, rep_pairs, cond_pairs = {}, 0, 0
        for pairs in all_maps.keys():
            if np.all(np.isnan(all_maps[pairs])): continue
            if maps_metadata[pairs[0]]==maps_metadata[pairs[1]]:
                rep_pairs +=1
                if "replicate" in all_maps_grouped: all_maps_grouped["replicate"]+= all_maps[pairs]
                else: all_maps_grouped["replicate"] = all_maps[pairs]
            else:
                cond_pairs +=1
                if "conditions" in all_maps_grouped: all_maps_grouped["conditions"]+= all_maps[pairs]
                else: all_maps_grouped["conditions"] = all_maps[pairs]
                    
        all_maps_grouped["replicate"]=all_maps_grouped["replicate"]/rep_pairs
        all_maps_grouped["conditions"]=all_maps_grouped["conditions"]/cond_pairs
        return all_maps_grouped
    
    def extract_features(self, chromosome, nfilter, pair, qthresh=0.999, min_length=10, max_length=256, min_width=10,max_width=256, pad_extra=3, im_size=20):
        if nfilter=="all": curr_map = np.concatenate([np.sum(value["replicate"][:,:int(self.pixel_size/2),:],axis=0) for chrom, value in self.paired_maps.items() 
                            if value is not None ], axis=1)
        else: curr_map = np.concatenate([value["replicate"][nfilter,:int(self.pixel_size/2),:] for chrom, value in self.paired_maps.items() 
                            if value is not None ], axis=1)

        pos_thresh = np.max([np.nanquantile(curr_map, qthresh),-np.nanquantile(curr_map, 1-qthresh)])
        neg_thresh = np.min([-np.nanquantile(curr_map, qthresh),np.nanquantile(curr_map, 1-qthresh)])

        if nfilter=="all": curr_map=np.sum(self.paired_maps[chromosome][pair][:,int(self.pixel_size/2):,:],axis=0)
        else: curr_map=self.paired_maps[chromosome][pair][nfilter,int(self.pixel_size/2):,:]

        arr_pos, arr_neg =  (curr_map>pos_thresh), (curr_map<neg_thresh)
        arr_pos, arr_neg =  ndimage.label(arr_pos), ndimage.label(arr_neg)

        features = []
        for pos_or_neg, arr in enumerate([arr_pos, arr_neg]):
            for feature_index in np.unique(arr[0])[1:]:
                indices=np.where(arr[0]==feature_index)
                x1, x2, y1, y2 = min(indices[0]), max(indices[0]), min(indices[1]), (max(indices[1])
                if ((x2-x1)<=min_length )|((y2-y1)<=min_width): continue 
                if ((x2-x1)>=max_length )|((y2-y1)>=max_width): continue  
                temp = convex_hull_image(arr[0]==feature_index)
                temp = temp[x1:x2, y1:y2]

                if (temp.shape[0]<=min_length )|(temp.shape[1]<=min_width): continue
                if (temp.shape[0]>=max_length )|(temp.shape[1]>=max_width): continue

                original_dims, height= temp.shape, np.min(indices[0])
                temp = resize(temp, (im_size,im_size),anti_aliasing=False,preserve_range=True)
                features.append((feature_index, temp, original_dims, height, arr[1],pos_or_neg, qthresh, [chromosome, x1,x2, y1, y2])
        return features

    def make_maps(self,chromosome, diagonal_off=4):
        all_maps = self.make_maps_base(chromosome, diagonal_off=diagonal_off)
        all_maps_grouped = self.make_maps_grouped(all_maps)
        return all_maps_grouped
        
