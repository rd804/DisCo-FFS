import numpy as np
import pickle
import sys
from disco_functions import *



class feature_loader():

	def __init__(self, feature_dict):
		self.dic = feature_dict
		self.keys = list(feature_dict.keys())

		print('loading features ....',self.keys)
	

	def efp_loader(self):
		variables = self.dic['efp']
		hettemp_efp = '/het/p1/ranit/qg/features/efp/'
		print(variables)
		for k in variables:
			if k==variables[0]:
				efp_test = np.load(hettemp_efp+'test/efp_'+str(k)+'.npy').reshape(-1,1)
			else:
				efp_temp =  np.load(hettemp_efp+'test/efp_'+str(k)+'.npy').reshape(-1,1)
				efp_test = np.hstack((efp_test,efp_temp))


		for k in variables:
			if k==variables[0]:
				efp_train = np.load(hettemp_efp+'train/efp_'+str(k)+'.npy').reshape(-1,1)
			else:
				efp_temp =  np.load(hettemp_efp+'train/efp_'+str(k)+'.npy').reshape(-1,1)
				efp_train = np.hstack((efp_train,efp_temp))


		for k in variables:
			if k==variables[0]:
				efp_val = np.load(hettemp_efp+'val/efp_'+str(k)+'.npy').reshape(-1,1)
			else:
				efp_temp =  np.load(hettemp_efp+'val/efp_'+str(k)+'.npy').reshape(-1,1)
				efp_val = np.hstack((efp_val,efp_temp))
		
		print(efp_train.shape)
		print(efp_val.shape)
		print(efp_test.shape)

		return efp_train,efp_val,efp_test



#	def bip_loader(self,basis='4_7'):
#		
#		variables = self.dic['bip']
#		hettemp_efp = '/het/p1/ranit/tops/bip/basis_4_7/'
		#variables = self.variables
#		print(variables)
#		for k in variables:
#			if k==variables[0]:
#				efp_test = np.load(hettemp_efp+'test/feature_'+str(k)+'.npy').reshape(-1,1)
#			else:
#				efp_temp =  np.load(hettemp_efp+'test/feature_'+str(k)+'.npy').reshape(-1,1)
#				efp_test = np.hstack((efp_test,efp_temp))


#		for k in variables:
#			if k==variables[0]:
#				efp_train = np.load(hettemp_efp+'train/feature_'+str(k)+'.npy').reshape(-1,1)
#			else:
#				efp_temp =  np.load(hettemp_efp+'train/feature_'+str(k)+'.npy').reshape(-1,1)
#				efp_train = np.hstack((efp_train,efp_temp))


#		for k in variables:
#			if k==variables[0]:
#				efp_val = np.load(hettemp_efp+'val/feature_'+str(k)+'.npy').reshape(-1,1)
#			else:
#				efp_temp =  np.load(hettemp_efp+'val/feature_'+str(k)+'.npy').reshape(-1,1)
#				efp_val = np.hstack((efp_val,efp_temp))
		
#		print(efp_train.shape)
#		print(efp_val.shape)
#		print(efp_test.shape)
#		return efp_train,efp_val,efp_test

	def m_loader(self):
		
		hettemp_efp = '/het/p1/ranit/qg/features/'
		m_train = np.load('{}mpt_train.npy'.format(hettemp_efp),allow_pickle=True)[:,0]
		m_val = np.load('{}mpt_val.npy'.format(hettemp_efp),allow_pickle=True)[:,0]
		m_test = np.load('{}mpt_test.npy'.format(hettemp_efp),allow_pickle=True)[:,0]

		return m_train.reshape(-1,1),m_val.reshape(-1,1),m_test.reshape(-1,1)

	
	def pt_loader(self):
				
		hettemp_efp = '/het/p1/ranit/qg/features/'
		pt_train = np.load('{}mpt_train.npy'.format(hettemp_efp),allow_pickle=True)[:,1]
		pt_val = np.load('{}mpt_val.npy'.format(hettemp_efp),allow_pickle=True)[:,1]
		pt_test = np.load('{}mpt_test.npy'.format(hettemp_efp),allow_pickle=True)[:,1]
		
		return pt_train.reshape(-1,1),pt_val.reshape(-1,1),pt_test.reshape(-1,1)


	def all_features(self):
		features = {}
		features['train']={}	
		features['val']={}	
		features['test']={}	
#		if 'bip' in self.keys:
#			name = 'bip'
#			features['train'][name],features['val'][name],features['test'][name] = self.bip_loader()
		if 'efp' in self.keys:
			name = 'efp'
			features['train'][name],features['val'][name],features['test'][name]= self.efp_loader()

		#initial = self.initial
		
		if 'm' in self.keys:
			name = 'm'	
			features['train'][name],features['val'][name],features['test'][name] = self.m_loader()
		if 'pt' in self.keys:
			name = 'pt'	
			features['train'][name],features['val'][name],features['test'][name] = self.pt_loader()
#		if 'mw' in self.keys:
#			name = 'mw'	
#			features['train'][name],features['val'][name],features['test'][name] = self.mw_loader()
		
		final_features = {}
		final_features['train'] = {}
		final_features['val'] = {}
		final_features['test'] = {}

		for i,key in enumerate(self.keys):
			if i==0:
				final_features['train']['first']=features['train'][key]
				final_features['val']['first']=features['val'][key]
				final_features['test']['first']=features['test'][key]

			else:
					
				final_features['train'][key]=features['train'][key]
				final_features['val'][key]=features['val'][key]
				final_features['test'][key]=features['test'][key]

		print(final_features['val'])
		stacked_features = {}
		stacked_features['train']=stack_features_dict(final_features['train'])
		stacked_features['val']=stack_features_dict(final_features['val'])
		stacked_features['test']=stack_features_dict(final_features['test'])


		return stacked_features
