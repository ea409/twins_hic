from HiCDataset import metriclearnpaired_HiCDataset, HiCDatasetDec
from metric_learn import SDML
import numpy as np
import pickle
import argparse
from reference_dictionaries import reference_genomes
import json

parser = argparse.ArgumentParser(description='metric learning baseline')
parser.add_argument('json_file',  type=str,
                    help='a file location for the json dictionary containing file paths')
parser.add_argument('outfile',  type=str,
                    help='a path for the output file')
parser.add_argument("data_name", type=str,help="key from dictionary containing paths for training and validation sets.")

args = parser.parse_args()

with open(args.json_file) as json_file:
    dataset = json.load(json_file)

data = [HiCDatasetDec.load(data_path) for data_path in dataset[args.data_name]["training"]]

mt = metriclearnpaired_HiCDataset(data, reference = reference_genomes[dataset[args.data_name]["reference"]], sims = [1,-1])
x= np.array(mt)
y= np.array(mt.labels)

sdml = SDML()
sdml.fit(x, y)

with open(args.outfile, 'wb') as output:
    output.write(pickle.dumps(sdml))
    output.close()