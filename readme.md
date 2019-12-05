# Hi-C Convolutional Neural Network
With data taken from .hic juicer dump files at 10kb, this repo produces small 880kb cleaned images which are partially overlapping from the diagonal on 
three biological phenotypes. The phenotypes are Wild Type, CTCF knockout and Rad21, CTCF double knockout and are all taken from mouse double positive thymocytes. 
The data is cleanded using split_files which is designed in order to minimize memory usage. The data can then bee loaded as a HiCclass dataset. 

The trained CNN is 78% accurate on the test chromosome (chr2), this data has been used in order to produce saliency maps by calling 

`from torch_plus import visualisation

GBP = visualisation.Guided(model) #visualisation.Vanilla(model) 
quickplot_all_reps(dataset,'chr2',j, GBP) `

The aim of this work is to provide biological insights on differences between phenotypes  

##Results 
The outcome is very clear distances identified by the classifier as being of interest. 

