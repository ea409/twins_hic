import torch 
from torch.utils.data import Dataset, DataLoader, RandomSampler
import sys
import os
from scipy.sparse import csr_matrix
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import cnn_main
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

model = cnn_main.Net(num_classes)
model.load_state_dict(torch.load('model.ckpt'))

transform = transforms.Compose([transforms.ToPILImage(),  transforms.ToTensor()])#,  transforms.Normalize([0.5], [0.5])])
metadata= pd.read_csv("cleaned_data_metadata.csv")
dataset= cnn_main.HiCDataset("cleaned_data", metadata, data_res, resolution, split_res, transform=transform)
sampler= RandomSampler(dataset,replacement=False)
dataloader = DataLoader(dataset, batch_size=len(dataset), sampler = sampler)

def prediction_plot(truth, pred, dataset):
    bools = ( (labels ==truth) & (predicted ==pred))
    indices = torch.nonzero(bools)
    rand = torch.randperm(len(indices))
    rand = rand[:20]
    fig = plot.figure(1, (20, 20))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(4, 5),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
    for i, ind in enumerate(indices[rand]): 
        img, lab = dataset[int(ind)]
        print(i)
        grid[i].imshow(np.array(img)[0])
    plot.show()
    return rand

# Test the model
 # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with model.eval():
    correct = 0
    total = 0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        outputs = F.softmax(model(inputs))
        total += len(labels)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()


prediction_plot(0, 0, dataset)
prediction_plot(0, 1, dataset)
prediction_plot(0, 2, dataset)
prediction_plot(1, 0, dataset)
prediction_plot(1, 1, dataset)
prediction_plot(1, 2, dataset)
prediction_plot(2, 0, dataset)
prediction_plot(2, 1, dataset)
prediction_plot(2, 2, dataset)



print('Test Accuracy of the model test images: {} %'.format(100 * correct / total))
