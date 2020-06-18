import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
#from torch_plus import additional_samplers
from HiCDataset import HiCDatasetDec, SiameseHiCDataset,GroupedHiCDataset
import models
import torch
from torch_plus.loss import ContrastiveLoss
import argparse

parser = argparse.ArgumentParser(description='Siamese network')
parser.add_argument('learning_rate',  type=float,
                    help='a float for the learning rate')
parser.add_argument('--batch_size',  type=int, default=17,
                    help='an int for batch size')
parser.add_argument('--epoch_training',  type=int, default=30,
                    help='an int for batch size')
parser.add_argument('--epoch_enforced_training',  type=int, default=0,
                    help='an int for batch size')
parser.add_argument('--outpath',  type=str, default="outputs/",
                    help='an int for batch size')

args = parser.parse_args()

cuda = torch.device("cuda:0")
hg19_dict = {'1': 249250621, '2': 243199373, '3': 198022430, '4': 191154276, '5': 180915260, '6': 171115067, '7': 159138663, '8': 146364022, '9': 141213431, '10': 135534747, '11': 135006516, '12': 133851895, '13': 115169878, '14': 107349540, '15': 102531392, '16': 90354753, '17': 81195210, '18': 78077248, '19': 59128983, '20': 63025520, '21': 48129895, '22': 51304566}
path = '/vol/bitbucket/ealjibur/data/'

#dataset all about
dataset = [ SiameseHiCDataset([HiCDatasetDec.load(path + "GSE113703_MDM_"+time+"_" + i + "_"+ j + ".mlhic" ) for i in ['mock', 'H5N1-UV','H5N1'] for j in ['r1','r2'] ],
             reference = ['hg19', hg19_dict] ) for time in ['6h', '12h','18h']]
Siamese = GroupedHiCDataset( dataset, reference ='hg19')
train_sampler = torch.utils.data.RandomSampler(Siamese)

#CNN params.
batch_size, learning_rate = args.batch_size, args.learning_rate
no_of_batches= np.floor(len(Siamese )/args.batch_size)
dataloader = DataLoader(Siamese, batch_size=args.batch_size, sampler = train_sampler)

#validation
dataset_validation = [ SiameseHiCDataset([HiCDatasetDec.load(path + "GSE113703_validation_MDM_"+time+"_" + i + "_"+ j + ".mlhic" ) for i in ['mock', 'H5N1-UV','H5N1'] for j in ['r1','r2'] ],
             reference = ['hg19', hg19_dict] ) for time in ['6h', '12h','18h']]
Siamese_validation  = GroupedHiCDataset( dataset_validation, reference ='hg19')
test_sampler = SequentialSampler(Siamese_validation)
batches_validation = np.ceil(len(dataset_validation)/100)
dataloader_validation = DataLoader(Siamese_validation, batch_size=100, sampler = test_sampler)

# Convolutional neural network (two convolutional layers)
model=models.SiameseNet().to(cuda)
model_save_path = args.outpath +'Siamese_nodrop_LR'+str(learning_rate)+'.ckpt'
torch.save(model.state_dict(),model_save_path)

# Loss and optimizer
criterion = ContrastiveLoss() #torch.nn.CosineEmbeddingLoss() #
optimizer = optim.Adagrad(model.parameters())

#  Training
for epoch in range(args.epoch_training):
    running_loss=0.0
    running_validation_loss = 0.0
    for i, data in enumerate(dataloader):
        input1, input2,  labels = data
        input1, input2 = input1.to(cuda), input2.to(cuda)
        labels = labels.type(torch.FloatTensor).to(cuda)
        # zero gradients
        optimizer.zero_grad()
        output1, output2 = model(input1, input2)
        loss = criterion(output1, output2, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % no_of_batches == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'
            .format(epoch+1, i, running_loss/no_of_batches))

    for i, data in enumerate(dataloader_validation):
        input1,  input2, labels = data
        input1, input2 = input1.to(cuda), input2.to(cuda)
        labels = labels.type(torch.FloatTensor).to(cuda)
        output1, output2 = model(input1, input2)
        loss = criterion(output1, output2, labels)
        running_validation_loss += loss.item()

    print ('Epoch [{}/{}], Loss: {:.4f}'
            .format(epoch+1, i, running_validation_loss/batches_validation ))
    if (epoch>args.epoch_enforced_training):
        prev_validation_loss = min(prev_validation_loss,running_validation_loss)
        if (float(prev_validation_loss) +  0.1 < float(running_validation_loss)):
            break
    else:
        prev_validation_loss =  running_validation_loss

    torch.save(model.state_dict(), model_save_path)





