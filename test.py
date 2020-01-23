import pandas as pd 
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from HiCDataset import HiCDataset, HiCType
import models
from matplotlib import pyplot as plt
#import torch_plus
# .visualisations when vis is done. 
from torch_plus import visualisation, additional_samplers

dataset=HiCDataset.load("HiCDataset_10kb_allreps_test")
test_sampler = SequentialSampler(dataset)

## for testing on only rad21 bound data from torch_plus import additional_samplers 
#indices_test=np.array(dataset.filter_by_Rad21("Cumulative_Rad21.txt", threshold=500))
#test_sampler = additional_samplers.SequentialSubsetSampler(indices_test)


#CNN params
batch_size, num_classes =100, 3

dataloader = DataLoader(dataset, batch_size=batch_size, sampler = test_sampler)  

model = models.DeepConvNet(num_classes)
model.load_state_dict(torch.load('models_final/model_10kb_deep_cnn_no_depth_adjustment.ckpt'))

# Test the model
# eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
correct = 0
total = 0
for i, data in enumerate(dataloader):
    data, depths = data
    inputs, labels =  data
    outputs = F.softmax(model(inputs), dim=1)
    total += len(labels)
    quality, predicted = torch.max(outputs.data, 1)
    correct += (predicted == labels).sum().item()
    if i % 5 == 0 :
        print('Test Accuracy of the model test images so far: {} %'.format(100 * correct / total))

print('Test Accuracy of the model test images: {} %'.format(100 * correct / total))

#vis.prediction_plot(-1, 0, inputs, labels, predicted, quality)
# GBP = visualisation.Guided(model) 
# for j in range(0,808):
#     fig=vis.quickplot_all_reps(dataset,'chr2',j, GBP)
#     fig.savefig('images/' + str(j)+'.png', dpi=fig.dpi, bbox_inches = 'tight')
#     plt.close()