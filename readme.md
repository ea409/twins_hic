# Hi-C Convolutional Neural Network
With data taken from .hic juicer dump files at 10kb, this repo produces small 880kb cleaned images which are partially overlapping from the diagonal on 
three biological phenotypes. The phenotypes are Wild Type, CTCF knockout and Rad21, CTCF double knockout and are all taken from mouse double positive thymocytes. 
The data is cleanded using split_files which is designed in order to minimize memory usage. The data can then bee loaded as a HiCclass dataset. 

The trained CNN is 78% accurate on the test chromosome (chr2), this data has been used in order to produce saliency maps by calling 

`from torch_plus import visualisation

GBP = visualisation.Guided(model) #visualisation.Vanilla(model) 
quickplot_all_reps(dataset,'chr2',j, GBP) `

The aim of this work is to provide biological insights on differences between phenotypes  

## Results 
The outcome is very clear distances identified by the classifier as being of interest. Left to right the images displayed show the saliency map, the HiC map and 
the saliency map overlayed onto the Hi-C map. Top to bottom the images are wild type, CTCF knockout and CTCF, Rad21 Knockout.

![](https://gitlab.doc.ic.ac.uk/ealjibur/CNN/output_example/Example_1.png)

This behaviour is replicable across many regions

| ![](https://gitlab.doc.ic.ac.uk/ealjibur/CNN/output_example/Example_2.png) | ![](https://gitlab.doc.ic.ac.uk/ealjibur/CNN/output_example/Example_3.png) |![](output_example/Example_4.png) |