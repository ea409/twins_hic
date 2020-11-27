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
from reference_dictionaries import reference_genomes
import json

parser = argparse.ArgumentParser(description='Siamese network')
parser.add_argument('model_name',  type=str,
                    help='a string indicating a model from models')
parser.add_argument('json_file',  type=str,
                    help='a file location for the json dictionary containing file paths')
parser.add_argument('learning_rate',  type=float,
                    help='a float for the learning rate')
parser.add_argument('--batch_size',  type=int, default=17,
                    help='an int for batch size')
parser.add_argument('--epoch_training',  type=int, default=30,
                    help='an int for no of epochs training can go on for')
parser.add_argument('--epoch_enforced_training',  type=int, default=0,
                    help='an int for number of epochs to force training for')
parser.add_argument('--outpath',  type=str, default="outputs/",
                    help='a path for the output directory')
parser.add_argument('--seed',  type=int, default=30004,
                    help='an int for the seed')
parser.add_argument('--mask',  type=bool, default=False,
                    help='an argument specifying if the diagonal should be masked')
parser.add_argument('--bias',  type=float, default=2,
                    help='an argument specifying the bias towards the contrastive loss function')
parser.add_argument("data_inputs", nargs='+',help="keys from dictionary containing paths for training and validation sets.")

args = parser.parse_args()

cuda = torch.device("cuda")

with open(args.json_file) as json_file:
    dataset = json.load(json_file)

torch.manual_seed(args.seed)

#dataset all about
Siamese = GroupedHiCDataset([ SiameseHiCDataset([HiCDatasetDec.load(data_path) for data_path in dataset[data_name]["training"]],
             reference = reference_genomes[dataset[data_name]["reference"]]) for data_name in args.data_inputs] )
train_sampler = torch.utils.data.RandomSampler(Siamese)

#CNN params.
batch_size, learning_rate = args.batch_size, args.learning_rate
no_of_batches= np.floor(len(Siamese)/args.batch_size)
dataloader = DataLoader(Siamese, batch_size=args.batch_size, sampler = train_sampler)

#validation
Siamese_validation  = GroupedHiCDataset([ SiameseHiCDataset([HiCDatasetDec.load(data_path) for data_path in dataset[data_name]["validation"]],
             reference = reference_genomes[dataset[data_name]["reference"]]) for data_name in args.data_inputs] )
test_sampler = SequentialSampler(Siamese_validation)
batches_validation = np.ceil(len(Siamese_validation)/100)
dataloader_validation = DataLoader(Siamese_validation, batch_size=100, sampler = test_sampler)

# Convolutional neural network (two convolutional layers)
model = eval("models."+ args.model_name)(mask=args.mask).to(cuda)
model_save_path = args.outpath + args.model_name  +'_' + str(learning_rate) +'_'+ str(batch_size)+'_' + str(args.seed) 
torch.save(model.state_dict(),model_save_path+'.ckpt')

#classification net 
nn_model = models.LastLayerNN().to(cuda)
torch.save(nn_model.state_dict(),model_save_path+"_nn.ckpt")

# Loss and optimizer
criterion = ContrastiveLoss() #tnn.CosineEmbeddingLoss() #
criterion2 = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters())

#  Training
for epoch in range(args.epoch_training):
    #training model
    running_loss = 0.0
    running_validation_loss = 0.0
    for i, data in enumerate(dataloader):
        input1, input2,  labels = data
        input1, input2 = input1.to(cuda), input2.to(cuda)
        labels = labels.to(cuda)
        # zero gradients
        optimizer.zero_grad()
        output1, output2 = model(input1, input2)
        output_class = nn_model(output1, output2)
        loss2 = criterion2(output_class, labels)
        labels = labels.type(torch.FloatTensor).to(cuda)
        loss1 = criterion(output1, output2, labels)
        loss = args.bias*loss1 + loss2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % no_of_batches == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'
            .format(epoch+1, i, running_loss/no_of_batches))
    #obtaining validation loss
    for i, data in enumerate(dataloader_validation):
        input1,  input2, labels = data
        input1, input2 = input1.to(cuda), input2.to(cuda)
        labels = labels.type(torch.FloatTensor).to(cuda)
        output1, output2 = model(input1, input2)
        loss = criterion(output1, output2, labels)
        running_validation_loss += loss.item()
    #evaluating model
    print ('Epoch [{}/{}], Loss: {:.4f}'
            .format(epoch+1, i, running_validation_loss/batches_validation ))
    if (epoch>args.epoch_enforced_training):
        prev_validation_loss = min(prev_validation_loss,running_validation_loss)
        if (float(prev_validation_loss) < 1.1*float(running_validation_loss)):
            break
    else:
        prev_validation_loss = running_validation_loss

    torch.save(model.state_dict(), model_save_path+'.ckpt')
    torch.save(nn_model.state_dict(),model_save_path+"_nn.ckpt")





