# Twins
This module contains an efficient data laoder and training structure for Hi-C data. The aim is to learn patterns from observations made along the diagonal of the Hi-C maps. This data is made up of reads which have typically been aligned, processed and normalised using either the HiCPro or the distiller pipeline. The resultant files of the format `.hic` or `.mcool` can be used as inputs for Twins. 

## Installation and Testing

### Requirements 
Python 3.6 or higher. Linux/OS. Separate installation of hic-straw is recommended. 

### Installation 
clone repository and then inside the repository run the following command: 

``` python setup.py install ```


## Load Hi-C Data 
Load Hi-C data from a `.hic` using the HiCDatasetDec type. Saving imediately after initialising is recommended since the initialisation is time and memory intensive.

```
from HiCDataset import HiCDatasetDec
replicate_id = 'R1' #replicate identifier can be anything  
class_id = 0 #class identifier must be int 
data = HiCDatasetDec(['input.hic', replicate_id , 'KR', 'BP', class_id],10000,880000)
data.save('input.mlhic')
```
Data saved as .mlhic can then be reloaded. Data from multiple Hi-C experiment files can be emalgamated into one sequential dataset or into a siamese dataset (i.e. paired by genomic location). Many siamese datasets or Hi-C datsets can be joined together. 

```
data_condition1 = HiCDatasetDec.load('input.mlhic')
data_condition2 = HiCDatasetDec.load('input2.mlhic')
data_condition3 = HiCDatasetDec.load('input3.mlhic')

from HiCDataset import GroupedHiCDataset
grouped_dataset = GroupedHiCDataset([data_condition1,data_condition2, data_condition3], reference = 'mm9')

from HiCDataset import SiameseHiCDataset
siamese1 = SiameseHiCDataset([data_condition1, data_condition2]) 
siamese2 = SiameseHiCDataset([data_condition1, data_condition3])
grouped = GroupedHiCDataset( [siamese1, siamese2], reference = 'mm9')

```

## Run Train Models 
Models can be trained using the siamese_train.py script. 

## Run Test Models

Models can be tested using the siamese_test.py script. Here a threshold is calculated using the train and validation for the separation of replicate pairs from condition pairs which is subsequentially used to calculate the percentage of replicate and condition pairs correctly identified. This script also produces two output plots one describing the train and validation distance distributions and the other the test. If the model has trained correctly without overfitting then these distibutions should be comparable. 

| Train |  Test |
| ------ | ------ |
| ![](output_example/train_dist.png)  | ![](output_example/test_dist.png)|

The distances between regions in their own right are of interest, regions with known differences in terms of enhancer activation etc are those which also have the highest euclidean distance - therefore using the euclidean distances or even looking at only regions with very high euclidean distances can help focus your research. 

## Feature Extraction 
An additional script for the generation of feature extraction maps are available in the feature extraction folder. 

## Downstream
Files demonstrating examples of downstream analysis can be found in the two jupyter labs notebooks.
