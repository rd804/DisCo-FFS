import pandas as pd
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
import sys



import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--scratch", help="start FS from scratch", action="store_true")
parser.add_argument('')
parser.add_argument("-tops", "--tops", help="Do FS on top-dataset", action="store_true")
parser.add_argument("-qg", "--qg", help="Do FS on qg-dataset",action="store_true")
parser.add_argument("--iter", help="iteration", type=int)
parser.add_argument("--exp_name", help="name unique to the run", type=str)



args = parser.parse_args()


# Save directory

if args.tops:
	save_dir = f'results/{args.exp_name}'

if args.qg:
	# TODO
	save_dir = '/het/p1/ranit/qg/disco_ffs/temp/'


# Save path for the iteration

iter_save_path = save_dir+'discor_'+str(args.exp_name)+'/iteration_'+str(args.iter)+'/'

if not args.scratch: 
	# TODO
	with open(f'{save_dir}features_{args.exp_name}.txt','rb') as fp:
		features = pickle.load(fp)

	with open(f'{save_dir}duplicate_features_{args.exp_name}.txt','rb') as fp:
		duplicate_features = pickle.load(fp)

# Load the features already selected

if args.scratch:
	if args.iter==0:
		features = {}
		duplicate_features= {}
	else:
		with open(f'{save_dir}features_{args.exp_name}.txt','rb') as fp:
			features = pickle.load(fp)

		with open(f'{save_dir}duplicate_features_{args.exp_name}.txt','rb') as fp:
			duplicate_features = pickle.load(fp)

print('features already selected: ',features)

# Create empty list for the new features
if args.iter==0:
	features['efp']=[]
	duplicate_features['efp']=[]

# Load a small batch of efp features, to check if the new feature has duplicates
if args.tops:
	efp_val_ = np.load("/het/p1/ranit/tops/data/efp_val_first10_7.5k_wjets.npy")

if args.qg:
	efp_val_ = np.load("/het/p1/ranit/qg/features/efp_batch_small.npy")

print(efp_val_.shape)

# Parse the disco scores for the iteration
files = [file for file in listdir(iter_save_path) if isfile(join(iter_save_path, file))]
list_files_numbers = [int(i.split('_')[2].split('.')[0]) for i in files]
list_files_numbers.sort()

print(len(list_files_numbers))

for index in list_files_numbers:
	path1=iter_save_path+'dis_cor_'+str(index)+'.txt'
	if index == list_files_numbers[0]:
		with open(path1,'rb') as fp:
			discor = pickle.load(fp)
	else:    
			
		with open(path1,'rb') as fp:
			discor_temp = pickle.load(fp)
		discor.extend(discor_temp)

no_efp = len(discor)
all_efps=7500

print('numer of feature scores computed:', no_efp)

# Assertion to check if all the features scores were computed
# Beware of condor errors
if no_efp < all_efps:
	print('fatal error: score for all features were not computed :'+str(no_efp)+' efps',file=sys.stderr)
	assert False

# Sort scores
dis = np.asarray(discor)
indices1 = (-dis).argsort()[:all_efps]
print('max disco: .......... ',np.amax(dis))
print('min disco: ........ ', np.amin(dis))


efps = 7500

# Add highest feature score to the list of selected features, ommit duplicates
for l in indices1:
		if l<efps:
			if l not in features['efp']:
				if l not in duplicate_features['efp']:
					new_feature = l
					features['efp'].append(new_feature)
					break
		else:
			if l not in features['mf_s2']:
#				if l not in duplicate_features['mf_s2']:
				new_feature = l
				features['mf_s2'].append(new_feature-efps)
				break

# If the new feature has duplicates, add them to the list of duplicate features
if new_feature<efps: 
	for efp_index in range(efps):
		diff = np.sum(efp_val_[:,efp_index] - efp_val_[:,new_feature])
		if diff==0:
			duplicate_features['efp'].append(efp_index)


# 
print('index selected ',new_feature)
print('feature added ', features)
print('duplicate features ',duplicate_features)


# Save the new list of features
with open(f'{save_dir}features_{args.exp_name}.txt','wb') as fp:
	pickle.dump(features,fp)

with open(f'{save_dir}duplicate_features_{args.exp_name}.txt','wb') as fp:
	pickle.dump(duplicate_features,fp)







