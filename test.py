from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from torch.autograd import Variable
import HiCclass
import models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd 
import numpy as np
import torch 

resolution = 440000
split_res = 4
data_res = 5000
transform = transforms.Compose([transforms.ToPILImage(),  transforms.ToTensor()])

metadata= pd.read_csv("~/documents/CNN/data/metadata.csv")
dataset=HiCclass.HiCDataset("~/documents/CNN/data/cleaned_data_low_res", metadata, data_res, resolution, split_res, transform=transform)
sampler= RandomSampler(dataset,replacement=False)
dataloader = DataLoader(dataset, batch_size=17, sampler = sampler)

model = models.Net(3)
model.load_state_dict(torch.load('~/documents/HiCinitial/data/JuicerDumpFIles/model_cnn_2layer.ckpt'))

def prediction_plot(truth, pred, inputs, labels, rand=None):
    if truth < 0: 
        bools = (predicted ==pred)
    else: 
        bools = ( (labels ==truth) & (predicted ==pred))
    indices = torch.nonzero(bools)
    if rand == None: 
        rand = torch.randperm(len(indices))
        rand = rand[:100]
    fig = plt.figure(1, (20, 20))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(10, 10),  # creates 2x2 grid of axes
                    axes_pad=0,  # pad between axes in inch.
                    )
    for i, ind in enumerate(indices[0:100]): 
        grid[i].imshow(np.array(inputs[int(ind)])[0])
    plt.show()
    return rand

# Test the model
 # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
correct = 0
total = 0
labels_all = []
pred_all = []
for i, data in enumerate(dataloader):
    inputs, labels = data
    outputs = F.softmax(model(inputs), dim=1)
    total += len(labels)
    labels_all.append(labels)
    _, predicted = torch.max(outputs.data, 1)
    pred_all.append(predicted)
    correct += (predicted == labels).sum().item()
    print(i/(102), correct, total)


prediction_plot(-1, 1, inputs, labels)

print('Test Accuracy of the model test images: {} %'.format(100 * correct / total))
