from HiCDataset import metriclearnpaired_HiCDataset
from metric_learn import SDML
import numpy as np
import pickle


mt = metriclearnpaired_HiCDataset(Siamese_validation, reference = reference_genomes[dataset[data_name]["reference"]], sims = [1,-1], resolution=880000, stride=4)
x= np.array(mt)
y= np.array(mt.labels)

sdml = SDML()
sdml.fit(x, y)

with open("sdml_trained", 'wb') as output:
    output.write(pickle.dumps(sdml))
    output.close()