import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
#from torch_plus import additional_samplers
from HiCDataset import  HiCDatasetDec, SiameseHiCDataset,GroupedHiCDataset
import models
import torch
from torch_plus.loss import ContrastiveLoss
import matplotlib.pyplot as plt
import argparse
from reference_dictionaries import reference_genomes
import json

parser = argparse.ArgumentParser(description='Siamese network testing module')
parser.add_argument('model_name',  type=str,
                    help='a string indicating a model from models')
parser.add_argument('json_file',  type=str,
                    help='a file location for the json dictionary containing file paths')
parser.add_argument('model_infile',  type=str,
                    help='a string indicating the model location file')
parser.add_argument("data_inputs", nargs='+',help="keys from dictionary containing paths for training and validation sets.")

args = parser.parse_args()


with open(args.json_file) as json_file:
    dataset = json.load(json_file)



def test_model(model, dataloader):
    distances = np.array([])
    labels = np.array([])
    for _, data in enumerate(dataloader):
        input1, input2, label = data
        input1, input2 = input1.to(cuda), input2.to(cuda)
        label = label.type(torch.FloatTensor).to(cuda)
        output1, output2 = model(input1, input2)
        predicted = F.pairwise_distance(output1,output2)
        distances = np.concatenate((distances, predicted.cpu().detach().numpy()))
        label = label.type(torch.FloatTensor)
        labels  = np.concatenate((labels, label.cpu().detach().numpy()))
    return distances, labels

cuda = torch.device("cuda:0")
model = eval("models."+ args.model_name)().to(cuda)
model.load_state_dict(torch.load(args.model_infile))
model.eval()

#dataset all
Siamese = GroupedHiCDataset([ SiameseHiCDataset([HiCDatasetDec.load(data_path) for data_path in (dataset[data_name]["training"]+ dataset[data_name]["validation"])],
             reference = reference_genomes[dataset[data_name]["reference"]]) for data_name in args.data_inputs] )
test_sampler = SequentialSampler(Siamese)
dataloader = DataLoader(Siamese, batch_size=100, sampler = test_sampler)

# Test the model
distances, labels = test_model(model, dataloader)

mx = max(distances)
mn = min(distances[distances>0])
rng = np.arange(mn, mx, (mx-mn)/200)

a = plt.hist(distances[(labels==0)],bins=rng, stacked=True, density=True, label='technical replicates')
b = plt.hist(distances[(labels==1)],bins=rng, stacked=True, density=True, label='biological conditions')
intersect = a[0][np.argwhere(np.diff(np.sign(a[0]-b[0])))[0]]
plt.axvline(intersect, color='k')
plt.xticks(np.arange(0,np.ceil(mx), 1))
plt.legend()
plt.title("distance of train and validation from "+ args.model_infile.split("/")[-1] +" on: "+ ", ".join(args.data_inputs))
plt.ylabel("density")
plt.xlabel("euclidean distance of representation")
plt.savefig(args.model_infile.split(".ckpt")[0]+"_train_distribution.pdf")
plt.close()

#dataset test
Siamese = GroupedHiCDataset([ SiameseHiCDataset([HiCDatasetDec.load(data_path) for data_path in (dataset[data_name]["test"])],
             reference = reference_genomes[dataset[data_name]["reference"]]) for data_name in args.data_inputs] )
test_sampler = SequentialSampler(Siamese)
dataloader = DataLoader(Siamese, batch_size=100, sampler = test_sampler)

# Test the model
distances, labels = test_model(model, dataloader)

global_rate = sum(((distances<intersect)==(labels==0)) )/len(distances)
TR_rate =  sum((distances<intersect) & (labels==0))/sum(labels==0)
BC_rate = sum((distances>intersect) & (labels==1) )/sum(labels==1)

print('global rate: {:.4f}, technical replicate rate: {:.4f}, biological condition rate: {:.4f}'
            .format(global_rate, TR_rate, BC_rate))

a = plt.hist(distances[(labels==0)],bins=rng, density=True, label='technical replicates')
b = plt.hist(distances[(labels==1)],bins=rng, density=True, label='biological conditions')
plt.axvline(intersect, color='k')
plt.xticks(np.arange(0,np.ceil(mx), 1))
plt.title("distance of test from " + args.model_infile.split("/")[-1] +" on: "+ ", ".join(args.data_inputs))
plt.ylabel("density")
plt.xlabel("euclidean distance of representation")
plt.legend()
plt.savefig(args.model_infile.split(".ckpt")[0]+"_test_distribution.pdf")
plt.close()
