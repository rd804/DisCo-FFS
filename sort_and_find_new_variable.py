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


#feature='bip'

j = args.j
m = args.m

if args.tops:
	hettemp = '/het/p1/ranit/tops/training_method_ypred_cut_04_06_changing_threshold/temp/'

if args.qg:
	hettemp = '/het/p1/ranit/qg/disco_ffs/temp/'

path = hettemp+'discor_'+str(j)+'/iteration_'+str(m)+'/'

if not args.scratch: 
	with open(hettemp+'variables_7k_iter_'+str(j)+'.txt','rb') as fp:
		variables = pickle.load(fp)

	with open(hettemp+'variables1_7k_iter_'+str(j)+'.txt','rb') as fp:
		variables1 = pickle.load(fp)

if args.scratch:
	if m==0:
		variables = {}
		variables1= {}
	else:
		with open(hettemp+'variables_7k_iter_'+str(j)+'.txt','rb') as fp:
			variables = pickle.load(fp)

		with open(hettemp+'variables1_7k_iter_'+str(j)+'.txt','rb') as fp:
			variables1 = pickle.load(fp)

print(variables)

if m==0:
#	variables['efp']=[]
#	variables1['efp']=[]

	variables['efp']=[]
	variables1['efp']=[]
	variables['mf_s2']=[]
	variables1['mf_s2']=[]

if args.tops:
	efp_val_ = np.load("/het/p1/ranit/tops/data/efp_val_first10_7.5k_wjets.npy")

if args.qg:
	efp_val_ = np.load("/het/p1/ranit/qg/features/efp_batch_small.npy")

print(efp_val_.shape)


files = [file for file in listdir(path) if isfile(join(path, file))]

# need to sort the files, to put them in correct order
list_files_numbers = [int(i.split('_')[2].split('.')[0]) for i in files]

list_files_numbers.sort()

print(len(list_files_numbers))

for i in list_files_numbers:
#       # although the directory is called discor, it could be any variable used for feature selection.
	path1=path+'dis_cor_'+str(i)+'.txt'
	if i == list_files_numbers[0]:
		with open(path1,'rb') as fp:
			discor = [pickle.load(fp)]
#			print(len(discor))
	else:    
#		df2 = pd.read_pickle("./python_logs/test/test_" +str(i)+".pkl")
			
		with open(path1,'rb') as fp:
			discor_temp = pickle.load(fp)
		discor.append(discor_temp)
	#	print(len(discor))

dis1 = list(np.concatenate(discor))
#print(dis1)
no_efp = len(dis1)
#all_efps = 7350
all_efps=9230

print('numer of features:',no_efp)
print(dis1)


if no_efp < all_efps:
	print('fatal error: score for all features were not computed :'+str(no_efp)+' efps',file=sys.stderr)

dis = np.asarray(dis1)
indices1 = (-dis).argsort()[:all_efps]
print('max disco: .......... ',np.amax(dis))
print('min disco: ........ ', np.amin(dis))


efps = 7500

for l in indices1:
		if l<efps:
			if l not in variables['efp']:
				if l not in variables1['efp']:
					var = l
					variables['efp'].append(var)
					break
		else:
			if l not in variables['mf_s2']:
#				if l not in variables1['mf_s2']:
				var = l
				variables['mf_s2'].append(var-efps)
				break

#for l in indices1:
#		if l<efps:
#			if l not in variables['efp']:
#				if l not in variables1['efp']:
#					var = l
#					variables['efp'].append(var)
#					break

#		elif (l==all_efps) & ('m' not in variables):
#			var = l
#			variables['m']=None
#			break
#		elif (l==all_efps+1) & ('pt' not in variables):
#			var = l
#			variables['pt']=None
#			break
#		elif (l==all_efps+2) & ('mw' not in variables):
#			var = l
#			variables['mw']=None
#			break

if var<efps: 
	for x in range(efps):
	#	diff = np.sum(efp_val_[x,:] - efp_val_[var,:])
		diff = np.sum(efp_val_[:,x] - efp_val_[:,var])
		if diff==0:
			variables1['efp'].append(x)


print('index selected ',var)
print('feature added ', variables)
print('duplicate features ',variables1)

m = len(variables)


with open(hettemp+'variables_7k_iter_'+str(j)+'.txt','wb') as fp:
	pickle.dump(variables,fp)

with open(hettemp+'variables1_7k_iter_'+str(j)+'.txt','wb') as fp:
	pickle.dump(variables1,fp)







