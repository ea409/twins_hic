import glob as glob 
import pandas as pd 
import numpy as np 
import sys
import os
import argparse

parser = argparse.ArgumentParser(description='File to clean juicer dump files into the correct format.')
parser.add_argument('to_dir',
                    help='the directory to save files into')
parser.add_argument('root_dir', 
                    help='file locations, include * for glob to work. e.g. ./*')
parser.add_argument('res',type=int, default=880000,  
                    help='resolution to be considered for each picture, current default is 880kb which is the median size of a TAD.')
parser.add_argument('split_res', type=int, default=8, 
                    help='number of overlaps per resolution sized window. Must be a divisor of resolution. current default is 8')

args = parser.parse_args()

try:
    to_dir = args.to_dir
except NameError:
    print("to directory not specified")
    exit()
    
try:
    root_dir = args.root_dir
except NameError:
    print("root directory not specified")
    exit()

resolution = args.res
split_res = args.split_res

if resolution % split_res != 0:
    print('split resolution is not a divisor of resolution.' )
    exit()


def split_files(to_dir, root_dir, resolution, split_res): 
    if not os.path.exists(to_dir):                  # caller handles errors
        os.mkdir(to_dir)                            # make dir, read/write parts
    else:
        for fname in os.listdir(to_dir):            # delete any existing files
            os.remove(os.path.join(to_dir, fname))
    index = 0
    metadata = pd.DataFrame()
    for no, filename in enumerate(glob.glob(root_dir)):
        first_index = index 
        df = pd.read_csv(filename, names=list(['x','y','vals']), usecols =['x', 'y', 'vals'], sep='\t')
        df = df.dropna()
        df = df.sort_values(['x', 'y'],ascending=[1,1])
        df = df.reset_index(drop=True)
        df = df.loc[np.abs(df.y -df.x) <  resolution]
        stop_cond = min(df.x)+resolution/split_res
        printfile = os.path.join(to_dir, str(no)+'_'+str(index))
        for i, row in df.iterrows():
            if row.x >stop_cond:
                stop_cond += resolution/split_res
                index +=1
                printfile  = os.path.join(to_dir, str(no)+'_'+str(index))
            f=open(printfile, 'a+') 
            f.write(str(int(row.x)) +'\t' + str(int(row.y)) +'\t' +str(row.vals) +'\n')
            f.close()
        index = index - split_res+1
        metadata = metadata.append(pd.DataFrame(data={'index':[no],'start':[min(df.x)], 'file': [filename], 'end': [index], 'first_index': [first_index]} ) )
    metadata=metadata.set_index('index')
    return metadata

metadata = split_files(to_dir, root_dir, resolution, split_res)

metadata.to_csv(to_dir+'/metadata.csv')