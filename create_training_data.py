import pandas as pd
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
import sys





import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--scratch", help="start FS from scratch", action="store_true")

parser.add_argument("-tops", "--tops", help="Do FS on top-dataset", action="store_true")
parser.add_argument("-qg", "--qg", help="Do FS on qg-dataset",action="store_true")
parser.add_argument("m", help="iteration", type=int)
parser.add_argument("j", help="name unique to the run", type=str)
#parser.add_argument("i", help="name unique to the run", type=str)


#parser.add_argument("square",help="squares the input of the file", type=int,choices = [0,1,2,3,4])

args = parser.parse_args()

if args.tops:
	hettemp = '/het/p1/ranit/tops/training_method_ypred_cut_04_06_changing_threshold/temp/'

	from feature_loader import *
#	initial_features = {'m':None, 'pt': None, 'mw': None, 'kin': None}
	initial_features = {'m':None, 'pt': None}

if args.qg:

	from feature_loader_qg import *
	hettemp = '/het/p1/ranit/qg/disco_ffs/temp/'
	initial_features = {'m':None, 'pt': None}

j = args.j
m = args.m


if not args.scratch:

	if m==0:
		
		variables = initial_features
		variables1 = initial_features
		
		with open(hettemp+'variables_7k_iter_'+str(j)+'.txt','wb') as fp:
			pickle.dump(variables,fp)

		with open(hettemp+'variables1_7k_iter_'+str(j)+'.txt','wb') as fp:
			pickle.dump(variables1,fp)
	else:

		with open(hettemp+'variables_7k_iter_'+str(j)+'.txt','rb') as fp:
			variables = pickle.load(fp)

		with open(hettemp+'variables1_7k_iter_'+str(j)+'.txt','rb') as fp:
			variables1 = pickle.load(fp)

else:

	with open(hettemp+'variables_7k_iter_'+str(j)+'.txt','rb') as fp:
		variables = pickle.load(fp)

	with open(hettemp+'variables1_7k_iter_'+str(j)+'.txt','rb') as fp:
		variables1 = pickle.load(fp)

print(variables)
print(variables1)


features = feature_loader(variables)
stacked_features = features.all_features()


np.save(hettemp+'features/train_'+str(m)+'_iter_'+str(j)+'.npy',stacked_features['train'])
np.save(hettemp+'features/test_'+str(m)+'_iter_'+str(j)+'.npy',stacked_features['test'])
np.save(hettemp+'features/val_'+str(m)+'_iter_'+str(j)+'.npy',stacked_features['val'])

if args.qg:
	trainlabels=np.load('/het/p1/ranit/qg/data/y_train.npy')
	vallabels=np.load('/het/p1/ranit/qg/data/y_val.npy')
	testlabels=np.load('/het/p1/ranit/qg/data/y_test.npy')

	
	np.save(hettemp+'features/trainlabels.npy',trainlabels)
	np.save(hettemp+'features/testlabels.npy',testlabels)
	np.save(hettemp+'features/vallabels.npy',vallabels)
