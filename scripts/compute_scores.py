import numpy as np
import time
import os
import pickle
import sys
import dcor
import random
import pandas as pd
from sklearn.utils import shuffle
import time
from src.utils import *
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--scratch", help="start FS from scratch", action="store_true")
parser.add_argument("-tops", "--tops", help="Do FS on top-dataset", action="store_true")
parser.add_argument("-qg", "--qg", help="Do FS on qg-dataset",action="store_true")
parser.add_argument("--feature", type=str, default='efp', help='features for which the score is computed')

parser.add_argument("--confusion_window_type", type = str, default='fixed', help='type of confusion window to select')
parser.add_argument("--high_threshold",type = float, default=0.7, help='the higher threshold for confusion window')
parser.add_argument("--low_threshold",type = float, default=0.3, help='the lower threshold for confusion window')


parser.add_argument("--parallel_index", help="parallel instance of score computation", type=int)
parser.add_argument("--parallel_step",default=2, help="number of features per parallel_index", type=int)

parser.add_argument("--iter", help="iteration", type=int)
parser.add_argument("--exp_name", type=str, help="name unique to the run")

args = parser.parse_args()

start_time = time.time()

# if path exists
save_dir = f'/het/p2/ranit/DisCo-FFS/results/{args.exp_name}'
feature = args.feature
save_index = args.parallel_index

if os.path.exists(f"{save_dir}/discor/iteration_"+str(args.iter)+"/dis_cor_"+str(save_index)+".txt"):
	sys.exit('already computed')

# Load labels

if args.tops:
	
	with open("data/y_train.txt", "rb") as fp:
		y_train = np.asarray(pickle.load(fp))
	with open("data/y_val.txt", "rb") as fp:
		y_val = np.asarray(pickle.load(fp))


	from src.feature_loader import *


if args.qg:
	# TODO
	from src.feature_loader_qg import *	
	
	hettemp = '/het/p1/ranit/qg/disco_ffs/temp/'
	if not os.path.exists(hettemp):
		os.makedirs(hettemp)

	y_train = np.load('/het/p1/ranit/qg/data/y_train.npy',allow_pickle=True)
	y_val = np.load('/het/p1/ranit/qg/data/y_val.npy',allow_pickle=True)

# Feature used for DisCo-FFS

#if j<375:
#	feature = 'efp'
#	save_index = args.parallel_index
#else:
#	feature = 'mf_s2'
#	save_index = args.parallel_index
#	args.parallel_index-=375

print('finished loading')


# feature indices for the parallel indices for score computation
if feature=='efp':
	start_var = args.parallel_index * args.parallel_step
	end_var = args.parallel_index * args.parallel_step + args.parallel_step

elif feature=='bip':
	start_var = args.parallel_index * args.parallel_step
	end_var = args.parallel_index * args.parallel_step + args.parallel_step

elif feature=='mf_s2':
	start_var = args.parallel_index * args.parallel_step
	end_var = args.parallel_index * args.parallel_step + args.parallel_step


if args.confusion_window_type == 'fixed':
	t_high = args.high_threshold
	t_low = args.low_threshold
else:
	pass
	# TODO

print('starting feature: ................... ', start_var)
print('ending feature: ................... ', end_var)

# Load features for score computation
feature_list = np.arange(start_var,end_var,1).tolist()
feature_dict = {feature: feature_list}
features = feature_loader(feature_dict)
feature_array = features.all_features()

# stack train and val
score_feature_train_val = np.vstack((feature_array['train'],feature_array['val']))
y_train_val = np.vstack((y_train.reshape(-1,1),y_val.reshape(-1,1)))

# Load already obtained features and classifier score
if not args.scratch:
	#TODO
	ypred=np.load(f'{save_dir}/ypred/ypred_{args.iter}.npy')
	print('ypred shape: ', ypred.shape)
	#ypred=ypred_batch	

	traindata = np.load(f'{save_dir}/features/train.npy')
	valdata = np.load(f'{save_dir}/features/val.npy')
	known_feature_train_val = np.vstack((traindata,valdata))
	

if args.scratch:

	# TODO
	if args.iter!=0:
		
		ypred_batch=np.load(hettemp+'ypred_batch/ypred_batch_'+str(args.iter)+'_'+str(args.exp_name)+'.npy')
		print('ypred_batch shape: ', ypred_batch.shape)
		ypred=ypred_batch
		
		traindata = np.load(hettemp+'features/train_'+str(args.iter)+'_iter_'+str(args.exp_name)+'.npy')
		valdata = np.load(hettemp+'features/val_'+str(args.iter)+'_iter_'+str(args.exp_name)+'.npy')
		known_feature_train_val = np.vstack((traindata,valdata))
		


print('in function')
print('Iteration ................. : ', args.iter)



assert len(score_feature_train_val) == len(y_train_val)\
== len(y_train) + len(y_val) == len(known_feature_train_val)
assert score_feature_train_val.shape[1] == args.parallel_step


# Select events within confusion window
if args.scratch:
	# TODO 
	if args.iter==0:
		score_feature_confusion = score_feature_train_val
		y_confusion = y_train_val
	
	if args.iter!=0:
		known_feature_confusion = known_feature_train_val[(ypred>t_low) & (ypred<t_high)]
		y_confusion = y_train_val[(ypred>t_low) & (ypred<t_high)]
		score_feature_confusion = score_feature_train_val[(ypred>t_low) & (ypred<t_high)]

if not args.scratch:
	known_feature_confusion = known_feature_train_val[(ypred>t_low) & (ypred<t_high)]
	y_confusion = y_train_val[(ypred>t_low) & (ypred<t_high)]
	score_feature_confusion = score_feature_train_val[(ypred>t_low) & (ypred<t_high)]
	
	# TODO shuffle

dis_cor_mean = []

print('score_feature_confusion shape: ', score_feature_confusion.shape)
print('known_feature_confusion shape: ', known_feature_confusion.shape)

assert len(known_feature_confusion) == len(y_confusion) == len(score_feature_confusion)


# Split the confusion window into minibatches
mini_batches = mini_batch_splitter(y_confusion,2048)
#################################################################################
##################################################################################	


# Compute score
for feature in range(args.parallel_step):
	
	if args.scratch:
		# TODO
		if args.iter==0:
			efp_mpt = score_feature_confusion[:,feature]
		else:
			efp_mpt=stack_features(first=known_feature_confusion,x=score_feature_confusion[:,feature])
	else:

		# Stack each feature for which the score is to be computed, 
		# with already known features
		 
		stacked_features=stack_features(first=known_feature_confusion,x=score_feature_confusion[:,feature])
	dcor_value = disco_mini_batch(stacked_features,y_confusion,mini_batches)
	print(dcor_value)
	dis_cor_mean.append(dcor_value) 

save_file = f"{save_dir}/discor/iteration_"+str(args.iter)+"/dis_cor_"+str(save_index)+".txt"
with open(save_file, 'wb') as fp:
	pickle.dump(dis_cor_mean, fp)



end_time = time.time()
print('time taken: ', end_time-start_time)	


