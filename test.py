import pandas as pd 
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import HiC_class
import models
#import torch_plus
# .visualisations when vis is done. 
from torch_plus import visualisation, additional_samplers

#Hi-C params
resolution, split_res, data_res = 880000, 8, 10000 #Old params: 440000, 4, 5000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToPILImage(),  transforms.ToTensor()])

metadata= pd.read_csv("10kb_allreps/metadata.csv")
dataset=HiC_class.HiCDataset("10kb_allreps", metadata, data_res, resolution, split_res, transform=transform)

indices_test = HiC_class.get_meta_index(dataset, ['chr2'],['WTR3','WTR4'])
test_sampler = additional_samplers.SequentialSubsetSampler(indices_test)

#CNN params
batch_size, num_classes =len(indices_test), 3

dataloader = DataLoader(dataset, batch_size=batch_size, sampler = test_sampler)  

model = models.ConvNet(num_classes)
model.load_state_dict(torch.load('../../model_10kb.ckpt'))

# Test the model
# eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
correct = 0
total = 0
for i, data in enumerate(dataloader):
    inputs, labels = data
    outputs = F.softmax(model(inputs), dim=1)
    total += len(labels)
    quality, predicted = torch.max(outputs.data, 1)
    correct += (predicted == labels).sum().item()
    if i % 5 == 0 :
        print('Test Accuracy of the model test images so far: {} %'.format(100 * correct / total))

print('Test Accuracy of the model test images: {} %'.format(100 * correct / total))

#vis.prediction_plot(-1, 0, inputs, labels, predicted, quality)

