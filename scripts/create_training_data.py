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

parser.add_argument("-s", "--scratch", help="start FS from scratch", action="store_true")
#parser.add_argument("--initial_features",type=ast.literal_eval, help = 'Initial features for DisCo-FFS')
parser.add_argument("-tops", "--tops", help="Do FS on top-dataset", action="store_true")
parser.add_argument("-qg", "--qg", help="Do FS on qg-dataset",action="store_true")
parser.add_argument("--iter", help="iteration", type=int)
parser.add_argument("--exp_name", help="name unique to the run", type=str)

args = parser.parse_args()

save_dir = f'/het/p1/ranit/DisCo-FFS/results/{args.exp_name}'
#initial_features = args.initial_features
initial_features = {'m':None, 'pt': None, 'mw': None}
if not os.path.exists(save_dir):
	os.makedirs(save_dir)


# This loads the feature loader, and creates initial features, if 
if not args.scratch:
	if args.tops:
		from src.feature_loader import *

	if args.qg:
		from src.feature_loader_qg import *

else:
	pass
	# TODO


# This allows the script to dump the initial features to a .txt file
# Or load already selected features

if not args.scratch:

	if args.iter==0:
		
		features = initial_features
		duplicate_features = initial_features
		
		with open(f'{save_dir}features.txt','wb') as fp:
			pickle.dump(features,fp)

		with open(f'{save_dir}duplicate_features.txt','wb') as fp:
			pickle.dump(duplicate_features,fp)
	else:

		with open(f'{save_dir}features.txt','rb') as fp:
			features = pickle.load(fp)

		with open(f'{save_dir}duplicate_features.txt','rb') as fp:
			duplicate_features = pickle.load(fp)

else:
	# TODO:

	pass



Features = feature_loader(features)
stacked_features = Features.all_features()

split = ['train','val','test']

if not os.path.exists(f'{save_dir}features'):
	os.makedirs(f'{save_dir}features')

for s in split:
	np.save(f'{save_dir}features/{s}.npy',stacked_features[s])

if args.qg:
	trainlabels=np.load('/het/p1/ranit/qg/data/y_train.npy')
	vallabels=np.load('/het/p1/ranit/qg/data/y_val.npy')
	testlabels=np.load('/het/p1/ranit/qg/data/y_test.npy')

	
	np.save(save_dir+'features/trainlabels.npy',trainlabels)
	np.save(save_dir+'features/testlabels.npy',testlabels)
	np.save(save_dir+'features/vallabels.npy',vallabels)
