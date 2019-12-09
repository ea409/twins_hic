from torch_plus import visualisation, additional_samplers
import numpy.ma as ma
import numpy as np
from skimage.filters import threshold_local
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.cm as cm 
import collections 

def get_positive_negative_saliency(gradient):
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency

def normalise_log_picture(picture,counts):
    for _ in range(counts):
        picture=np.log(picture+1)
        picture=picture/np.max(np.abs(picture))
    return picture

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

my_cm = cm.get_cmap('Reds')
my_cm._init()
my_cm._lut[:,-1] = np.concatenate( (np.zeros(120) ,np.linspace(0.0000001, 1, len(my_cm._lut[:,-1])-120) ) ) 

def quickplot(dataset,index, tc, method,cm1=my_cm,cm2 ='viridis'):
    pred_img = dataset[index].unsqueeze(1)
    pred_img.requires_grad = True
    pic=method.generate_gradients(pred_img, tc)
    fig = plt.figure()
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(1, 4),  # creates 2x2 grid of axes
                    axes_pad=0,  # pad between axes in inch.
                    )
    pos, _ = get_positive_negative_saliency(pic)
    pos = (pos[0]+np.transpose(pos[0]))/2
    pos_plot=threshold_local(pos,7)
    pos_plot=pos_plot/np.max(np.abs(pos_plot))
    file_plot=normalise_log_picture(dataset[index][0].numpy(),4)
    grid[0].imshow(pos_plot, 'coolwarm')
    grid[1].imshow(np.tril(pos_plot)+np.triu(file_plot),'coolwarm')
    grid[2].imshow(threshold_local(file_plot,5),cm2)
    grid[3].imshow(threshold_local(file_plot,5),cm2)
    grid[3].imshow(pos_plot,cmap=cm1)
    return pos, fig

def pos_saliency(dataset,index, tc, method):
    pred_img = dataset[index][0].unsqueeze(1)
    pic=method.generate_gradients(pred_img, tc)
    pos,_ =get_positive_negative_saliency(pic)
    pos = (pos[0]+np.transpose(pos[0]))/2
    pos =threshold_local(pos,7)
    pos =pos/np.max(np.abs(pos))
    return pos

def make_CTCF_map(dataset,metind, GBP):
    CTCF_predictions = np.zeros(11*(dataset.metadata.end[metind]-dataset.metadata.first_index[metind]+7), dtype=float)
    counts = np.zeros(11*(dataset.metadata.end[metind]-dataset.metadata.first_index[metind]+7), dtype=float)
    for i, indices in enumerate(range(dataset.metadata.first_index[metind],dataset.metadata.end[metind])):
        if i % 8 ==0:
            pos = pos_saliency(dataset,indices, 0, GBP)
            length=int((88-len(np.diagonal(pos,offset=4)))/2) 
            curr = np.concatenate((np.diagonal(pos,offset=4), np.zeros(length)))
            counts[(11*i+length):(11*i+88)] += np.concatenate(( np.ones(88-2*length), np.zeros(length)))
            CTCF_predictions[(11*i+length):(11*i+88)]+=curr
    return CTCF_predictions/counts

def quickplot_all_reps(dataset,chrom,inds,method,cm1=my_cm,cm2 ='viridis'):
    mets = dataset.metadata[dataset.metadata.file.str.contains(chrom)&dataset.metadata.file.str.contains('R1R2')].copy()
    mets = mets.sort_values(by='classification')
    fig = plt.figure()
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                    axes_pad=0,  # pad between axes in inch.
                    )
    fig.text(1-0.5,1-0.05,chrom +'   '+str(inds) )
    txt=''
    for i, met in enumerate(mets.iterrows()): 
        _, met = met 
        index = met.first_index+inds
        pos_plot = pos_saliency(dataset, index, i, method)
        file_plot=normalise_log_picture(dataset[index][0][0].numpy(),4)
        grid[0+3*i].imshow(pos_plot, 'coolwarm')
        grid[1+3*i].imshow(threshold_local(file_plot,5),cm2)
        grid[2+3*i].imshow(threshold_local(file_plot,5),cm2)
        grid[2+3*i].imshow(pos_plot,cmap=cm1)
        outputs=F.softmax(method.model(dataset[index][0].unsqueeze(1)), dim=1).detach().numpy()[0]
        txt = txt +'class ' +str(i)+": "+ '{0:.2f}, {1:.2f}, {2:.2f}      '.format(*outputs)
    fig.text(0.0,0.0,txt)
    return fig


