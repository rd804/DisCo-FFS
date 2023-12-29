import os
import pickle
import sys
#import dcor
import random
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import time

from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten,Reshape,Activation,ActivityRegularization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler, StandardScaler
import matplotlib.pyplot as plt
import pickle
from src.utils import *

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--scratch", help="start FS from scratch", action="store_true")
parser.add_argument("-tops", "--tops", help="Do FS on top-dataset", action="store_true")
parser.add_argument("-qg", "--qg", help="Do FS on qg-dataset",action="store_true")
parser.add_argument("--iter", help="iteration", type=int)
parser.add_argument("--exp_name", help="name unique to the run", type=str)
parser.add_argument("--epochs", help="number of epochs", type=int)
#parser.add_argument("square",help="squares the input of the file", type=int,choices = [0,1,2,3,4])

args = parser.parse_args()

iter = args.iter
exp_name = args.exp_name

if args.tops:
    with open("/home/rd804/DisCo-FFS/data/y_train.txt", "rb") as fp:
        trainlabels = np.asarray(pickle.load(fp))
    with open("/home/rd804/DisCo-FFS/data/y_val.txt", "rb") as fp:
        vallabels = np.asarray(pickle.load(fp))
    with open("/home/rd804/DisCo-FFS/data/y_test.txt", "rb") as fp:
        testlabels = np.asarray(pickle.load(fp))
    
   # hettemp = '/home/rd804/ypred_method/temp/'

if args.qg:
    
    hettemp = '/home/rd804/qg/disco_ffs/temp/'
    trainlabels = np.load('{}features/trainlabels.npy'.format(hettemp))
    vallabels = np.load('{}features/vallabels.npy'.format(hettemp))
    testlabels = np.load('{}features/testlabels.npy'.format(hettemp))

os.environ["CUDA_VISIBLE_DEVICES"]= "2"
#variables = indices1
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  tf.config.experimental.set_memory_growth(physical_devices[1], True)
  tf.config.experimental.set_memory_growth(physical_devices[2], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

#m=6

    

if not args.scratch:
	if iter==0:
	    epochs=50
	else:
	   
	    epochs=args.epochs
  

if args.scratch:
	epochs=args.epochs 

save_dir = f'/home/rd804/DisCo-FFS/results/{args.exp_name}/'


traindata = np.load(f'{save_dir}features/train.npy')
valdata = np.load(f'{save_dir}features/val.npy')
testdata = np.load(f'{save_dir}features/test.npy')

alldata = np.vstack((traindata,valdata))
num_classes = 2
batch_size=512

scaler = RobustScaler()

scaler.fit(alldata)
traindata = scaler.transform(traindata)
valdata = scaler.transform(valdata)
testdata = scaler.transform(testdata) 

alldata = scaler.transform(alldata)


print('....... training data ..... : ',traindata.shape)
print('....... all data ..... : ',alldata.shape)    

ytrain = tf.keras.utils.to_categorical(trainlabels, num_classes)
yval = tf.keras.utils.to_categorical(vallabels, num_classes)
ytest = tf.keras.utils.to_categorical(testlabels, num_classes)

# model = Sequential()
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(traindata.shape[-1],)))
#  model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.1))


# model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

#checkpoint_filepath_auc = '/home/rd804/ypred_method/model/checkpoint_'+str(m)+'_'+str(i)
if not os.path.exists(f'{save_dir}model'):
    os.makedirs(f'{save_dir}model')

checkpoint_filepath_auc = f'{save_dir}model/checkpoint_{iter}'

model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001))
#model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate =0.0001),metrics=[tf.keras.metrics.SpecificityAtSensitivity(0.3,name='r30')])

model_checkpoint_callback_auc = tf.keras.callbacks.ModelCheckpoint(
filepath=checkpoint_filepath_auc, 
save_weights_only=True,
monitor='val_loss',
mode='auto',
save_best_only=True,min_delta=1e-10)   

history = model.fit(traindata, ytrain,
                batch_size=batch_size,epochs=epochs,
                verbose=0, workers=6,
                validation_data=(valdata, yval),callbacks=[model_checkpoint_callback_auc])


model.load_weights(checkpoint_filepath_auc)
ypred=model.predict(testdata)[:,1]

#end=time.time()
#print(end-start,'time taken')
_,fpr30,_ = threshold_30(testlabels,ypred,0.3)
_,fpr50,_ = threshold_30(testlabels,ypred,0.5)
auc = roc_auc_score(testlabels,ypred)

r30 = 1/fpr30
r50 = 1/fpr50 

metrics = ['auc','r30','r50']
metric_values = [auc,r30,r50]

for i,metric in enumerate(metrics):
    if not os.path.exists(f'{save_dir}results/{metric}'):
        os.makedirs(f'{save_dir}results/{metric}')
    
    np.save(f'{save_dir}results/{metric}/{metric}_{iter}.npy',metric_values[i])


#np.save(f'{save_dir}auc_{exp_name}_{iter}.npy',auc)
#np.save(f'{save_dir}r30_{exp_name}_{iter}.npy',r30)
#np.save(f'{save_dir}r50_{exp_name}_{iter}.npy',r50)

model.load_weights(checkpoint_filepath_auc)
ypred_=model.predict(alldata)[:,1]
print(ypred[0:20])

if not os.path.exists(f'{save_dir}ypred'):
    os.makedirs(f'{save_dir}ypred')

np.save(f'{save_dir}ypred/ypred_{iter}.npy',ypred_)

if not os.path.exists(f'{save_dir}loss'):
    os.makedirs(f'{save_dir}loss')

#np.save('./r30_list_method0_20/r30_'+str(m)+'.npy',np.asarray([r30]))
print(r30)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right', frameon=False)
plt.savefig(f'{save_dir}loss/loss_{iter}.jpg')
plt.close()




#traindata = stack_features(first = m_train,x=pt_train,z=top_wmass['train'],w=efp_train,y=tau32_train)
#testdata = stack_features(first = m_test,x=pt_test,z=top_wmass['test'],w=efp_test,y=tau32_test)
#valdata = stack_features(first = m_val,x=pt_val,z=top_wmass['val'],w=efp_val,y=tau32_val)

