import pandas as pd
import numpy as np
import pickle
import os
from os import listdir
from os.path import isfile, join
import sys
import ast




import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-tops", "--tops", help="Do FS on top-dataset", action="store_true")
parser.add_argument("--iter", help="iteration", type=int)
parser.add_argument("--exp_name", help="name unique to the run", type=str)

args = parser.parse_args()

save_dir = f'/het/p2/ranit/DisCo-FFS/results/{args.exp_name}'
#initial_features = args.initial_features
initial_features = {'m':None, 'pt': None, 'mw': None}
if not os.path.exists(save_dir):
	os.makedirs(save_dir)


# This loads the feature loader, and creates initial features, if 
if args.tops:
	from src.feature_loader import *


# This allows the script to dump the initial features to a .txt file
# Or load already selected features


if args.iter==0:
	
	features = initial_features
	duplicate_features = initial_features
	
	with open(f'{save_dir}/features.txt','wb') as fp:
		pickle.dump(features,fp)

	with open(f'{save_dir}/duplicate_features.txt','wb') as fp:
		pickle.dump(duplicate_features,fp)
else:

	with open(f'{save_dir}/features.txt','rb') as fp:
		features = pickle.load(fp)

	with open(f'{save_dir}/duplicate_features.txt','rb') as fp:
		duplicate_features = pickle.load(fp)



# loads and saves already selected/initial features
		
Features = feature_loader(features)
stacked_features = Features.all_features()

split = ['train','val','test']

if not os.path.exists(f'{save_dir}/features'):
	os.makedirs(f'{save_dir}/features')

for s in split:
	np.save(f'{save_dir}/features/{s}.npy',stacked_features[s])
