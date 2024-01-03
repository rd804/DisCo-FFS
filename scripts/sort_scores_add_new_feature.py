import pandas as pd
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
import sys
import os
from matplotlib import pyplot as plt

# TODO: Add wanb logging


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--feature',default = 'efp' ,type = str, help = 'the features on which disco-ffs is applied on')
parser.add_argument("-tops", "--tops", help="Do FS on top-dataset", action="store_true")
parser.add_argument("--iter", help="iteration", type=int)
parser.add_argument("--exp_name", help="name unique to the run", type=str)



args = parser.parse_args()


# Save directory

if args.tops:
	save_dir = f'results/{args.exp_name}'

# Save path for disco values

disco_path = f'{save_dir}/discor/iteration_{args.iter}/'

# Load the features already selected


with open(f'{save_dir}/features.txt','rb') as fp:
	features = pickle.load(fp)

with open(f'{save_dir}/duplicate_features.txt','rb') as fp:
	duplicate_features = pickle.load(fp)


print('features already selected: ',features)


# Create empty list for the new features
if args.iter==0:
	features[args.feature]=[]
	duplicate_features[args.feature]=[]

# Load a small batch of efp features, to check if the new feature has duplicates
if args.tops:
	efp_val_ = np.load("/het/p1/ranit/tops/data/efp_val_first10_7.5k_wjets.npy")

if args.qg:
	efp_val_ = np.load("/het/p1/ranit/qg/features/efp_batch_small.npy")

print(efp_val_.shape)

# Parse the disco scores for the iteration
files = [file for file in listdir(disco_path) if isfile(join(disco_path, file))]
list_files_numbers = [int(i.split('_')[2].split('.')[0]) for i in files]
list_files_numbers.sort()

print(len(list_files_numbers))

for index in list_files_numbers:
	path1=disco_path+'dis_cor_'+str(index)+'.txt'
	if index == list_files_numbers[0]:
		with open(path1,'rb') as fp:
			discor = pickle.load(fp)
	else:    
			
		with open(path1,'rb') as fp:
			discor_temp = pickle.load(fp)
		discor.extend(discor_temp)

no_efp = len(discor)
all_efps=7350

print('number of feature scores computed:', no_efp)

# Assertion to check if all the features scores were computed
# Beware of condor errors
if no_efp < all_efps:
	print('fatal error: score for all features were not computed :'+str(no_efp)+' efps',file=sys.stderr)
	#assert False

# Sort scores
dis = np.asarray(discor)
indices1 = (-dis).argsort()[:all_efps]
print('max disco: .......... ',np.amax(dis))
print('min disco: ........ ', np.amin(dis))



# Add highest feature score to the list of selected features, ommit duplicates
for l in indices1:
	if l not in features[args.feature]:
		if l not in duplicate_features[args.feature]:
			new_feature = l
			features[args.feature].append(new_feature)
			break


# If the new feature has duplicates, add them to the list of duplicate features
for efp_index in range(all_efps):
	diff = np.sum(efp_val_[:,efp_index] - efp_val_[:,new_feature])
	if diff==0:
		duplicate_features[args.feature].append(efp_index)


# 
print('index selected ',new_feature)
print('feature added ', features)
print('duplicate features ',duplicate_features)


# Save the new list of features
with open(f'{save_dir}/features.txt','wb') as fp:
	pickle.dump(features,fp)

with open(f'{save_dir}/duplicate_features.txt','wb') as fp:
	pickle.dump(duplicate_features,fp)

# plot feature histogram vs confusion window feature histogram
if args.tops:
	from src.feature_loader import *


Feature = feature_loader(features)
stacked_features = Feature.all_features()
with open("data/y_train.txt", "rb") as fp:
	y_train = np.asarray(pickle.load(fp))


if not os.path.exists(f'{save_dir}/features_plots'):
	os.makedirs(f'{save_dir}/features_plots')

if not os.path.exists(f'{save_dir}/features_plots/iter_{args.iter}'):
	os.makedirs(f'{save_dir}/features_plots/iter_{args.iter}')

ypred=np.load(f'{save_dir}/ypred/ypred_{args.iter}.npy')[0:len(y_train)]
y_confusion = y_train[(ypred>0.3) & (ypred<0.7)]


for f in range(stacked_features['train'].shape[1]):

	feature_array = stacked_features['train'][:,f]
	feature_confusion = stacked_features['train'][(ypred>0.3) & (ypred<0.7),f]
	# bins within some percentile
	bins = np.linspace(np.percentile(feature_array,0.5),np.percentile(feature_array,99.5),100)
	#plt.figure()
	plt.hist(feature_array[y_train==0],bins=bins,label='feature bkg',histtype='stepfilled', alpha = 0.5, density=True)
	plt.hist(feature_array[y_train==1],bins=bins,label='feature sig', histtype='stepfilled', alpha = 0.5,density=True)
	plt.hist(feature_confusion[y_confusion==0],bins=bins,label='feature bkg confusion',histtype='step',density=True)
	plt.hist(feature_confusion[y_confusion==1],bins=bins,label='feature sig confusion', histtype='step',density=True)
	plt.legend()
	plt.savefig(f'{save_dir}/features_plots/iter_{args.iter}/feature_{f}.png')
	plt.close()

plt.hist(dis,bins=100,label='disco',histtype='stepfilled')
plt.savefig(f'{save_dir}/features_plots/iter_{args.iter}/disco_distribution.png')
plt.close()





