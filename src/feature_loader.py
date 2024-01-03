import numpy as np
import pickle
import sys
from src.utils import *



class feature_loader():

	def __init__(self, feature_dict):
		self.dic = feature_dict
		self.keys = list(feature_dict.keys())

		print('loading features ....',self.keys)
	

	def efp_loader(self):
		features = self.dic['efp']
		#hettemp_efp = '/het/p1/ranit/tops/data/efp_7k_log/'
		hettemp_efp = 'data/tops/efp/'
		print(features)
		for k in features:
			if k==features[0]:
				efp_test = np.load(hettemp_efp+'test/efp_'+str(k)+'.npy').reshape(-1,1)
			else:
				efp_temp =  np.load(hettemp_efp+'test/efp_'+str(k)+'.npy').reshape(-1,1)
				efp_test = np.hstack((efp_test,efp_temp))


		for k in features:
			if k==features[0]:
				efp_train = np.load(hettemp_efp+'train/efp_'+str(k)+'.npy').reshape(-1,1)
			else:
				efp_temp =  np.load(hettemp_efp+'train/efp_'+str(k)+'.npy').reshape(-1,1)
				efp_train = np.hstack((efp_train,efp_temp))


		for k in features:
			if k==features[0]:
				efp_val = np.load(hettemp_efp+'val/efp_'+str(k)+'.npy').reshape(-1,1)
			else:
				efp_temp =  np.load(hettemp_efp+'val/efp_'+str(k)+'.npy').reshape(-1,1)
				efp_val = np.hstack((efp_val,efp_temp))
		
		print(efp_train.shape)
		print(efp_val.shape)
		print(efp_test.shape)

		return efp_train,efp_val,efp_test


	def m_loader(self):
		
		with open("/het/p4/ranit/data/mlist_test.txt", "rb") as fp:
			m_test = np.asarray(pickle.load(fp))
		with open("/het/p4/ranit/data/mlist_train.txt", "rb") as fp:
			m_train = np.asarray(pickle.load(fp))
		with open("/het/p4/ranit/data/mlist_val.txt", "rb") as fp:
			m_val = np.asarray(pickle.load(fp))     

		return m_train.reshape(-1,1),m_val.reshape(-1,1),m_test.reshape(-1,1)

	
	def pt_loader(self):
				
		with open("/het/p4/ranit/data/pTlist_train.txt", "rb") as fp:
			pt_train = np.asarray(pickle.load(fp))
		with open("/het/p4/ranit/data/pTlist_val.txt", "rb") as fp:
			pt_val = np.asarray(pickle.load(fp))
		with open("/het/p4/ranit/data/pTlist_test.txt", "rb") as fp:
			pt_test = np.asarray(pickle.load(fp))

		return pt_train.reshape(-1,1),pt_val.reshape(-1,1),pt_test.reshape(-1,1)

	def mw_loader(self):
		with open('/het/p4/ranit/data/top_wmass.txt','rb') as fp:
			wmass = pickle.load(fp)
	
		return wmass['train'].reshape(-1,1),wmass['val'].reshape(-1,1),wmass['test'].reshape(-1,1)
	

	def all_features(self):
		features = {}
		features['train']={}	
		features['val']={}	
		features['test']={}	

		if 'efp' in self.keys:
			name = 'efp'
			if len(self.dic[name])>0:
				name = 'efp'
				features['train'][name],features['val'][name],features['test'][name]= self.efp_loader()


		if 'm' in self.keys:
			name = 'm'	
			features['train'][name],features['val'][name],features['test'][name] = self.m_loader()
		if 'pt' in self.keys:
			name = 'pt'	
			features['train'][name],features['val'][name],features['test'][name] = self.pt_loader()
		if 'mw' in self.keys:
			name = 'mw'	
			features['train'][name],features['val'][name],features['test'][name] = self.mw_loader()
		
		final_features = {}
		final_features['train'] = {}
		final_features['val'] = {}
		final_features['test'] = {}

		for i,key in enumerate(self.keys):
			if self.dic[key] is not None:
				if len(self.dic[key])>0:
					if i==0:
						final_features['train']['first']=features['train'][key]
						final_features['val']['first']=features['val'][key]
						final_features['test']['first']=features['test'][key]

					else:
							
						final_features['train'][key]=features['train'][key]
						final_features['val'][key]=features['val'][key]
						final_features['test'][key]=features['test'][key]

			if self.dic[key] is None:
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
