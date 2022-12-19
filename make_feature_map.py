from torch.utils.data import Dataset, DataLoader, SequentialSampler
from HiSiNet.HiCDatasetClass import HiCDataset, HiCDatasetDec, SiameseHiCDataset, PairOfDatasets
from HiSiNet.reference_dictionaries import reference_genomes
from HiSiNet import models
import numpy as np
import sys
import pandas as pd
import torch
import argparse



parser = argparse.ArgumentParser(description='Siamese network')
parser.add_argument('model_name',  type=str,
                    help='a string indicating a model from models')
parser.add_argument('model_file',  type=str,
                    help='a path for the model')
parser.add_argument('save_path',  type=str,
                    help='a path to save the feature map')
parser.add_argument('reference_genome',  type=str,
                    help='a string describing the reference')
parser.add_argument("data_inputs", nargs='+',help="keys from dictionary containing paths for training and validation sets.")

args = parser.parse_args()


data_paths =[data_path for data_path in data_inputs]
model = eval("models."+args.model_name)(mask=True)
model.load_state_dict(torch.load(args.model_file, map_location=torch.device('cpu')))
model.eval()

list_of_HiCDatasets=[HiCDatasetDec.load(data_path) for data_path in data_paths] 

Paired_map = PairOfDatasets(list_of_HiCDatasets, model, reference = reference_genomes[args.reference_genome])

del Paired_map.data 
del Paired_map.labels

Paired_map.save(args.save_path)
