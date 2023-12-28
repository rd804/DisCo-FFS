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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LeakyReLU
from sklearn.preprocessing import RobustScaler, StandardScaler
import matplotlib.pyplot as plt
import pickle
from disco_functions import *

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
    with open("/home/rd804/Tops/EFP/data/y_train.txt", "rb") as fp:
        trainlabels = np.asarray(pickle.load(fp))
    with open("/home/rd804/Tops/EFP/data/y_val.txt", "rb") as fp:
        vallabels = np.asarray(pickle.load(fp))
    with open("/home/rd804/Tops/EFP/data/y_test.txt", "rb") as fp:
        testlabels = np.asarray(pickle.load(fp))
    
    hettemp = '/home/rd804/ypred_method/temp/'

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
	if m==0:
	    epochs=50
	else:
	   
	    epochs=args.epochs
  

if args.scratch:
	epochs=args.epochs 


traindata = np.load('{}features/train_'.format(hettemp)+str(m)+'_iter_'+str(i)+'.npy')
valdata = np.load('{}features/val_'.format(hettemp)+str(m)+'_iter_'+str(i)+'.npy')
testdata = np.load('{}features/test_'.format(hettemp)+str(m)+'_iter_'+str(i)+'.npy')

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

checkpoint_filepath_auc = '/home/rd804/ypred_method/model/checkpoint_'+str(m)+'_'+str(i)
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate =0.001))
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

np.save('/home/rd804/ypred_method/auc/auc_'+str(m)+'_'+str(i)+'.npy',auc)
np.save('/home/rd804/ypred_method/r30/r30_'+str(m)+'_'+str(i)+'.npy',r30)
np.save('/home/rd804/ypred_method/r50/r50_'+str(m)+'_'+str(i)+'.npy',r50)

model.load_weights(checkpoint_filepath_auc)
ypred_=model.predict(alldata)[:,1]
print(ypred[0:20])

np.save('{}ypred_batch/ypred_batch_'.format(hettemp)+str(m)+'_'+str(i)+'.npy',ypred_)



#np.save('./r30_list_method0_20/r30_'+str(m)+'.npy',np.asarray([r30]))
print(r30)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('/home/rd804/ypred_method/loss_curves/loss_'+str(m)+'_'+str(i)+'.jpg')
plt.close()




#traindata = stack_features(first = m_train,x=pt_train,z=top_wmass['train'],w=efp_train,y=tau32_train)
#testdata = stack_features(first = m_test,x=pt_test,z=top_wmass['test'],w=efp_test,y=tau32_test)
#valdata = stack_features(first = m_val,x=pt_val,z=top_wmass['val'],w=efp_val,y=tau32_val)

