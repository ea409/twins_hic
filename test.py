from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from torch.autograd import Variable
import HiCclass
import models
import torch.nn.functional as F
import pandas as pd 
import numpy as np
import torch 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

#Hi-C params
resolution, split_res, data_res = 880000, 8, 10000 #440000, 4, 5000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToPILImage(),  transforms.ToTensor()])

metadata= pd.read_csv("10kb_allreps/metadata.csv")
dataset=HiCclass.HiCDataset("10kb_allreps", metadata, data_res, resolution, split_res, transform=transform)

indices_test = HiCclass.get_meta_index(metadata, ['chr2'], train=False)
test_sampler =  torch.utils.data.SubsetRandomSampler(indices_test)

#CNN params
batch_size, num_classes, learning_rate =17, 3, 0.2

dataloader = DataLoader(dataset, batch_size=min(len(indices_test),1500), sampler = test_sampler) #change to split test and train and add on thee other dataset. 

model = models.ConvNet(num_classes)
model.load_state_dict(torch.load('model_10kb.ckpt'))

def prediction_plot(truth, pred, inputs, labels, predicted, quality,rand=None):
    if truth < 0: 
        bools = (predicted==pred) 
    else: 
        bools = ( (labels ==truth) & (predicted ==pred))
    indices = torch.nonzero(bools)
    if rand != None: 
        rand = torch.randperm(len(indices))
        rand = rand[:100]
        ind2 = rand #randomizes the indices -if you want to select random ones instead of most.
    else:
        _, ind2  = torch.sort(-quality[indices], dim=0)
    fig = plt.figure(1, (20, 20))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(5, 5),  # creates 2x2 grid of axes
                    axes_pad=0,  # pad between axes in inch.
                    )
    for i, ind in enumerate(indices[ind2][0:25]): 
        grid[i].imshow(inputs[int(ind)].numpy()[0])
    plt.show()
    return indices[ind2][0:25]

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
    if i % 5000 == 0 :
        print(i/(102), correct, total)


prediction_plot(-1, 1, inputs, labels, predicted, quality)

print('Test Accuracy of the model test images: {} %'.format(100 * correct / total))
