import os
import pickle
import sys
import dcor
import random
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import time
from sklearn.preprocessing import RobustScaler, StandardScaler
import pickle
import torch
#from hsic import *

def stack_features_dict(dictionary):
    for i,j in dictionary.items():
        j = np.array(j)
        if j.ndim==1:
            j = j.reshape(-1,1)
           # print(j)
        if i=='first':
            array=j
           # print(j)
        else:
            array = np.hstack((array,j))
    print(array.shape)
    return array


def stack_features(**kwargs):
    for i,j in kwargs.items():
        j = np.array(j)
        if j.ndim==1:
            j = j.reshape(-1,1)
           # print(j)
        if i=='first':
            array=j
           # print(j)
        else:
            array = np.hstack((array,j))
    print(array.shape)
    return array
        
def stack_data(**kwargs):
    for i,j in kwargs.items():
        j = np.array(j)
        if j.ndim==1:
            j = j.reshape(-1,1)
           # print(j)
        if i=='first':
            array=j
         #   print(j)
        else:
            array = np.vstack((array,j))
    print(array.shape)
    return array        





def threshold_30(y_true,y_pred,thr): 
    #to find the threshold that gives TPR of 50%, and the false positive rate at a TPR of 50% 
    fpr_, tpr_, thresholds_ = roc_curve(y_true, y_pred)
    #print(tpr_)
    #print()
    t50 = thresholds_[np.argmin(np.absolute(tpr_-thr))]
    #print(t50)
    #print(t50)
    fpr50 = fpr_[thresholds_==t50].item()
    tpr50 = tpr_[thresholds_==t50].item()
    if fpr50>0:
        print('R30 at tpr ', tpr50, ' is ',1/fpr50)
    
    #print(tpr50)
    return  t50, fpr50, tpr50

def threshold_30_b(y_true,y_pred,thr): 
    #to find the threshold that gives TPR of 50%, and the false positive rate at a TPR of 50% 
    fpr_, tpr_, thresholds_ = roc_curve(y_true, y_pred) 
    #print(tpr_)
    #print() 
    t50 = thresholds_[np.argmin(np.absolute(fpr_-thr))]   
    #print(t50)
    #print(t50) 
    fpr50 = fpr_[thresholds_==t50].item()
    #print(fpr50)
    tpr50 = tpr_[thresholds_==t50].item()
    # print(tpr50)
    return  t50, fpr50, tpr50

def disco_mini_batch(vector1,vector2,minibatch_split):
    dis_cor = []
    for indices in minibatch_split:
        vector_1_mini_batch = vector1[indices]
        vector_2_mini_batch = vector2[indices]
        dis = dcor.distance_correlation_af_inv(vector_1_mini_batch,vector_2_mini_batch)
        dis_cor.append(dis)
    return np.mean(dis_cor)

def disco_mini_batch_partial(vector1,vector2,vector3,minibatch_split,normalize='True'):
    dis_cor = []
    if normalize == 'True':
        vector1 = vector1.reshape(-1,1)
        vector2 = vector2.reshape(-1,1)
       # vector3 = vector3.reshape(-1,1)
        
        scaler = RobustScaler()
        scaler.fit(vector1)
        vector1 = scaler.transform(vector1)
        
        scaler = RobustScaler()
        scaler.fit(vector2)
        vector2 = scaler.transform(vector2)
        
        scaler = RobustScaler()
        scaler.fit(vector3)
        vector3 = scaler.transform(vector3)
    #    print(np.amax(vector3))
     #   print(np.amin(vector3))

    for indices in minibatch_split:
        vector_1_mini_batch = vector1[indices]
        vector_2_mini_batch = vector2[indices]
        vector_3_mini_batch = vector3[indices]
        dis = dcor.partial_distance_correlation(vector_1_mini_batch,vector_2_mini_batch,vector_3_mini_batch)
        dis_cor.append(dis)
    return np.mean(dis_cor)



def mini_batch_splitter(data,batch_size,no_mini_batch='all',method='none'):
    #for data in data:
    total_shape = len(data)
    total_possible_mini_batches = total_shape//batch_size

 #   print(indices.shape)
    
    if no_mini_batch == 'all':
        total_mini_batches = total_possible_mini_batches
    else:
        total_mini_batches = no_mini_batch
 #   print(total_mini_batches)
    if method=='none':
        indices = np.array([i for i in range(total_shape)])
        for batch_no in range(total_mini_batches):
          #  print(batch_no)
            if batch_no == 0:
                data_mini_batch = indices[batch_no*batch_size:batch_no*batch_size+batch_size]
                #
            else:
                data_temp = indices[batch_no*batch_size:batch_no*batch_size+batch_size]
                data_mini_batch = np.vstack((data_mini_batch,data_temp))
            
    elif method =='random sample':
        for batch_no in range(total_mini_batches):
            if batch_no==0:
                data_mini_batch = np.array(random.sample(data.tolist(),batch_size))
            else:
                data_temp = np.array(random.sample(data.tolist(),batch_size))
                data_mini_batch = np.vstack((data_mini_batch,data_temp))
                
            
    return data_mini_batch
