#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from disco_functions import *
from efp_loader import *


# In[2]:


with open("/het/p4/ranit/data/mlist_test.txt", "rb") as fp:
    m_test = np.asarray(pickle.load(fp))
with open("/het/p4/ranit/data/mlist_train.txt", "rb") as fp:
    m_train = np.asarray(pickle.load(fp))
with open("/het/p4/ranit/data/mlist_val.txt", "rb") as fp:
    m_val = np.asarray(pickle.load(fp))     


with open("/het/p4/ranit/data/pTlist_train.txt", "rb") as fp:
    pt_train = np.asarray(pickle.load(fp))
with open("/het/p4/ranit/data/pTlist_val.txt", "rb") as fp:
    pt_val = np.asarray(pickle.load(fp))
with open("/het/p4/ranit/data/pTlist_test.txt", "rb") as fp:
    pt_test = np.asarray(pickle.load(fp))
    
    
with open("/het/p4/ranit/data/y_train.txt", "rb") as fp:
    y_train = np.asarray(pickle.load(fp))
with open("/het/p4/ranit/data/y_val.txt", "rb") as fp:
    y_val = np.asarray(pickle.load(fp))
with open("/het/p4/ranit/data/y_test.txt", "rb") as fp:
    y_test = np.asarray(pickle.load(fp))


# In[3]:


efp_train, efp_val,_ = efp_loader([4436, 4491, 3407, 1500, 4303, 2063])


# In[4]:


mpt_train_ = np.transpose(np.vstack((m_train,pt_train)))
mpt_val_ = np.transpose(np.vstack((m_val,pt_val)))

with open('/het/p4/ranit/data/top_wmass.txt','rb') as fp:
    wmass = pickle.load(fp)


mpt_train = np.hstack((mpt_train_,wmass['train'].reshape(-1,1)))
mpt_val = np.hstack((mpt_val_,wmass['val'].reshape(-1,1)))


# In[5]:


alldata = stack_data(first=mpt_train, x=mpt_val)


# In[6]:


ydisco = stack_data(first=y_train,x=y_val)


# In[19]:





# In[8]:


ypred = np.load('./temp/ypred_batch/ypred_batch_0_c4_efps_32_32.npy')



# In[39]:


def disco_minibatches(vector1,vector2,minibatch_split):
    dis_cor = []
    for indices in minibatch_split:
        vector_1_mini_batch = vector1[indices]
        vector_2_mini_batch = vector2[indices]
        dis = dcor.distance_correlation_af_inv(vector_1_mini_batch,vector_2_mini_batch)
        dis_cor.append(dis)
    return np.array(dis_cor)


# In[40]:





# In[45]:




# In[38]:


#disco_mini_batch(ydisco,ypred,minibatches)


# In[48]:




# In[9]:

print('increasing window size in ypred')
#thresh = 0.1
for thresh in [0.1,0.2,0.3,0.4,0.5]:
    ydisco_cut = ydisco[(ypred>0.5-thresh)&(ypred<0.5+thresh)]
    alldata_cut = alldata[(ypred>0.5-thresh)&(ypred<0.5+thresh)]

    minibatches = mini_batch_splitter(ydisco_cut,2048)
    disco = disco_mini_batch(ydisco_cut,alldata_cut,minibatches)
    print(disco)


# In[12]:


print('scanning window size of 0.2')
for thresh in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
    ydisco_cut = ydisco[(ypred>thresh)&(ypred<thresh+0.2)]
    alldata_cut = alldata[(ypred>thresh)&(ypred<thresh+0.2)]

    minibatches = mini_batch_splitter(ydisco_cut,2048)
    disco = disco_mini_batch(ydisco_cut,alldata_cut,minibatches)
    print(disco)


# In[11]:


print('scanning window size of 0.3')
for thresh in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
    ydisco_cut = ydisco[(ypred>thresh)&(ypred<thresh+0.3)]
    alldata_cut = alldata[(ypred>thresh)&(ypred<thresh+0.3)]

    minibatches = mini_batch_splitter(ydisco_cut,2048)
    disco = disco_mini_batch(ydisco_cut,alldata_cut,minibatches)
    print(disco)


# In[10]:


print('scanning window size of 0.4')
for thresh in [0,0.1,0.2,0.3,0.4,0.5,0.6]:
    ydisco_cut = ydisco[(ypred>thresh)&(ypred<thresh+0.4)]
    alldata_cut = alldata[(ypred>thresh)&(ypred<thresh+0.4)]

    minibatches = mini_batch_splitter(ydisco_cut,2048)
    disco = disco_mini_batch(ydisco_cut,alldata_cut,minibatches)
    print(disco)


# In[56]:


#thresh=0.2
#np.count([np.argwhere((ypred>0.5-thresh)&(ypred<0.5+thresh)).flatten()]


# In[ ]:




