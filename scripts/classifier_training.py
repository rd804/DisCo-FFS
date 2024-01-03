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

parser.add_argument("-tops", "--tops", help="Do FS on top-dataset", action="store_true")
parser.add_argument("--iter", help="iteration", type=int)
parser.add_argument("--exp_name", help="name unique to the run", type=str)
parser.add_argument("--epochs" ,default=500,help="number of epochs", type=int)
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
    

os.environ["CUDA_VISIBLE_DEVICES"]= "2"
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  tf.config.experimental.set_memory_growth(physical_devices[1], True)
  tf.config.experimental.set_memory_growth(physical_devices[2], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

 

if iter==0:
    epochs=50
else: 
    epochs=args.epochs


save_dir = f'/home/rd804/DisCo-FFS/results/{args.exp_name}/'

# load features
traindata = np.load(f'{save_dir}features/train.npy')
valdata = np.load(f'{save_dir}features/val.npy')
testdata = np.load(f'{save_dir}features/test.npy')

alldata = np.vstack((traindata,valdata))
num_classes = 2
batch_size=512

# preprocessing
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

## model ###############################
# model = Sequential()
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(traindata.shape[-1],)))
#  model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


if not os.path.exists(f'{save_dir}model'):
    os.makedirs(f'{save_dir}model')

checkpoint_filepath_auc = f'{save_dir}model/checkpoint_{iter}'

model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001))

model_checkpoint_callback_auc = tf.keras.callbacks.ModelCheckpoint(
filepath=checkpoint_filepath_auc, 
save_weights_only=True,
monitor='val_loss',
mode='auto',
save_best_only=True,min_delta=1e-10)   

# train model
history = model.fit(traindata, ytrain,
                batch_size=batch_size,epochs=epochs,
                verbose=0, workers=6,
                validation_data=(valdata, yval),callbacks=[model_checkpoint_callback_auc])

# evaluate model on testdata and save metrics
model.load_weights(checkpoint_filepath_auc)
ypred=model.predict(testdata)[:,1]

_,fpr30,_ = threshold_30(testlabels,ypred,0.3)
_,fpr50,_ = threshold_30(testlabels,ypred,0.5)
auc = roc_auc_score(testlabels,ypred)

r30 = 1/fpr30
r50 = 1/fpr50 

metrics = ['auc','r30','r50']
metric_values = [auc,r30,r50]

for i,metric in enumerate(metrics):
    if not os.path.exists(f'{save_dir}{metric}'):
        os.makedirs(f'{save_dir}{metric}')
    
    np.save(f'{save_dir}{metric}/{metric}_{iter}.npy',metric_values[i])


# save classifier prediction

model.load_weights(checkpoint_filepath_auc)
ypred_=model.predict(alldata)[:,1]
print(ypred[0:20])

if not os.path.exists(f'{save_dir}ypred'):
    os.makedirs(f'{save_dir}ypred')

np.save(f'{save_dir}ypred/ypred_{iter}.npy',ypred_)

if not os.path.exists(f'{save_dir}loss'):
    os.makedirs(f'{save_dir}loss')

print(r30)

# save loss plot
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

