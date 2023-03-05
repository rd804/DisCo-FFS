import numpy as np
#from matplotlib import pyplot as plt
import time
import os
import pickle
import sys
import dcor
import random
import pandas as pd
from sklearn.utils import shuffle
import time
#from efp_loader import efp_loader
from disco_functions import *
#from modified_do import * 

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--scratch", help="start FS from scratch", action="store_true")
parser.add_argument("-tops", "--tops", help="Do FS on top-dataset", action="store_true")
parser.add_argument("-qg", "--qg", help="Do FS on qg-dataset",action="store_true")
parser.add_argument("j", help="parallel instance of score computation", type=int)
parser.add_argument("m", help="iteration", type=int)
parser.add_argument("i", help="name unique to the run", type=str)
#parser.add_argument("square",help="squares the input of the file", type=int,choices = [0,1,2,3,4])

args = parser.parse_args()

j = args.j
m = args.m
i = args.i

start_time = time.time()


if args.tops:
	hettemp = '/het/p1/ranit/tops/training_method_ypred_cut_04_06_changing_threshold/temp/'
	with open("/het/p4/ranit/data/y_train.txt", "rb") as fp:
		y_train = np.asarray(pickle.load(fp))
	with open("/het/p4/ranit/data/y_val.txt", "rb") as fp:
		y_val = np.asarray(pickle.load(fp))
	with open("/het/p4/ranit/data/y_test.txt", "rb") as fp:
		y_test = np.asarray(pickle.load(fp))

	from feature_loader import *


if args.qg:
	from feature_loader_qg import *	
	
	hettemp = '/het/p1/ranit/qg/disco_ffs/temp/'
	if not os.path.exists(hettemp):
		os.makedirs(hettemp)

	y_train = np.load('/het/p1/ranit/qg/data/y_train.npy',allow_pickle=True)
	y_val = np.load('/het/p1/ranit/qg/data/y_val.npy',allow_pickle=True)
	y_test = np.load('/het/p1/ranit/qg/data/y_test.npy',allow_pickle=True)
#nsubjettiness_train =np.load('/scratch/rd804/data/nsubjettiness_8_train.npy') 
#nsubjettiness_test =np.load('/scratch/rd804/data/nsubjettiness_8_test.npy') 
#nsubjettiness_val =np.load('/scratch/rd804/data/nsubjettiness_8_val.npy') 

#tau3202_train = np.divide(nsubjettiness_train[:,15],nsubjettiness_train[:,14]+10**(-10))
#tau3202_val = np.divide(nsubjettiness_val[:,15],nsubjettiness_val[:,14]+10**(-10))
#tau3202_test = np.divide(nsubjettiness_test[:,15],nsubjettiness_test[:,14]+10**(-10))

feature = 'bip'
save_index = j




#if j<375:
#	feature = 'efp'
#	save_index = j
#else:
#	feature = 'mf_s2'
#	save_index = j
#	j-=375


#mpt_train = np.hstack((mpt_train_,tau3202_train.reshape(-1,1)))
#mpt_val = np.hstack((mpt_val_,tau3202_val.reshape(-1,1)))

y_disco = np.vstack((y_train.reshape(-1,1),y_val.reshape(-1,1)))
print('finished loading')


def find_next_variable(j,m,i):
	
	#print(ypred)	
	#if not os.path.exists(hettemp+"/discor_"+str(i)):
	#	os.mkdir(hettemp+"/discor_"+str(i))
#	if not os.path.exists(hettemp+"/discor_"+str(i)+"/iteration_"+str(m)):
#		os.mkdir(hettemp+"/discor_"+str(i)+"/iteration_"+str(m))	
	
	if feature=='efp':
		start_var = j*20
		end_var = j*20+20
	
	elif feature=='bip':
		start_var = j*3
		end_var = j*3+3
	
	elif feature=='mf_s2':
		start_var = j*10
		end_var = j*10+10

	t_high=0.7
	t_low=0.3	
	print('starting variable: ................... ',start_var)

	if not args.scratch:
		
		ypred_batch=np.load(hettemp+'ypred_batch/ypred_batch_'+str(m)+'_'+str(i)+'.npy')
		print('ypred_batch shape: ', ypred_batch.shape)
		ypred=ypred_batch
		
		#if m==0:

		#	traindata = mpt_train
		#	valdata = mpt_val
		#	alldata = np.vstack((traindata,valdata))

		#else:
		#	efp_train = np.load(hettemp+'efp/efp_train_'+str(m)+'_iter_'+str(i)+'.npy')
		#	efp_val = np.load(hettemp+'efp/efp_val_'+str(m)+'_iter_'+str(i)+'.npy')
			

		traindata = np.load(hettemp+'features/train_'+str(m)+'_iter_'+str(i)+'.npy')
		valdata = np.load(hettemp+'features/val_'+str(m)+'_iter_'+str(i)+'.npy')
		alldata = np.vstack((traindata,valdata))
		

	if args.scratch:

		
		if m!=0:
			
			ypred_batch=np.load(hettemp+'ypred_batch/ypred_batch_'+str(m)+'_'+str(i)+'.npy')
			print('ypred_batch shape: ', ypred_batch.shape)
			ypred=ypred_batch
			
			traindata = np.load(hettemp+'features/train_'+str(m)+'_iter_'+str(i)+'.npy')
			valdata = np.load(hettemp+'features/val_'+str(m)+'_iter_'+str(i)+'.npy')
			alldata = np.vstack((traindata,valdata))
			
			#efp_train = np.load(hettemp+'efp/efp_train_'+str(m)+'_iter_'+str(i)+'.npy')
			#efp_val = np.load(hettemp+'efp/efp_val_'+str(m)+'_iter_'+str(i)+'.npy')

			#traindata=efp_train
			#valdata = efp_val
			#alldata = np.vstack((traindata,valdata))	
		
		#else:	
		#	variable_list = np.arange(start_var,end_var,1).tolist()
		#	feature_dict = {'m':None, 'pt': None, 'mw': None}

		#	features = feature_loader(feature_dict)
		#	data = features.all_features()
		#	traindata
	#if j<500:	
	variable_list = np.arange(start_var,end_var,1).tolist()
	#feature_dict = {'efp': variable_list}
	feature_dict = {feature: variable_list}
	features = feature_loader(feature_dict)
	efp = features.all_features()
	
	#efp_train_,efp_val_,_=efp_loader(variable_list)	
	efp_disco=np.vstack((efp['train'],efp['val']))

	print('in function')
	print('value of m ................. : ', m)
	
	print('new efps shape: ',efp_disco.shape)
	#print('old training set shape ',alldata.shape)
	
	if args.scratch:
		if m==0:
			efp_disco_cut = efp_disco
			y_disco_cut = y_disco
		
		if m!=0:
			alldata_cut = alldata[(ypred>t_low) & (ypred<t_high)]
			y_disco_cut = y_disco[(ypred>t_low) & (ypred<t_high)]
			efp_disco_cut = efp_disco[(ypred>t_low) & (ypred<t_high)]
	
	if not args.scratch:
		alldata_cut = alldata[(ypred>t_low) & (ypred<t_high)]
		y_disco_cut=y_disco[(ypred>t_low) & (ypred<t_high)]
		efp_disco_cut = efp_disco[(ypred>t_low) & (ypred<t_high)]

	dis_cor_mean = []

#	t_high,_,_ = threshold_30(y_disco,ypred,0.90)
#	print('high threshold: ',t_high)	
	#print()
#	t_low,_,_ = threshold_30_b(y_disco,ypred,0.10)
#	print('low threshold: ',t_low)	
	#print(ypred)		
	
#	print(alldata_cut.shape)	
	print(y_disco_cut.shape)
	mini_batches = mini_batch_splitter(y_disco_cut,2048)
#################################################################################
##################################################################################	

	for k in range(efp_disco.shape[1]):

######################################################################################
######################################################################################
####### Insert your variable selection method here ##################################
# Create a function that takes 2 vectors as inputs and spits out a value for the method you
# are using. It could be disco(vector1,vector2,minibatches). Feed this value for each efp to 
# dis_cor_efp, and all this gets saved in the dis_cor_mean list.	
		if args.scratch:
			if m==0:
				efp_mpt = efp_disco_cut[:,k]
			else:
				efp_mpt=stack_features(first=alldata_cut,x=efp_disco_cut[:,k])
		#dcor=disco_mini_batch_partial(efp_disco_cut[:,k],y_disco_cut,alldata_cut,mini_batches)
		else:
			efp_mpt=stack_features(first=alldata_cut,x=efp_disco_cut[:,k])
		dcor=disco_mini_batch(efp_mpt,y_disco_cut,mini_batches)
		print(dcor)
		dis_cor_mean.append(dcor) 

	save_file = hettemp+"discor_"+str(i)+"/iteration_"+str(m)+"/dis_cor_"+str(save_index)+".txt"
	with open(save_file, 'wb') as fp:
		pickle.dump(dis_cor_mean, fp)



end_time = time.time()
print('time taken: ', end_time-start_time)	

    

if __name__ == "__main__":
    #m = int(float(sys.argv[2]))
    #j = int(float(sys.argv[1]))
    #i = int(float(sys.argv[3]))
    #i = str(sys.argv[3])
    
#    t_high = int(float(sys.argv[4]))
#    t_low = int(float(sys.argv[5]))
    find_next_variable(j,m,i)

