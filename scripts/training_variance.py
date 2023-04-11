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

from src.utils import *
#from modified_do import * 
import argparse

parser = argparse.ArgumentParser()

	#split = int(float(sys.argv[1]))
	#variable = int(float(sys.argv[2]))
	#name = str(sys.argv[3])
parser.add_argument("-s", "--scratch", help="start FS from scratch", action="store_true")
parser.add_argument("-tops", "--tops", help="Do FS on top-dataset", action="store_true")
parser.add_argument("-qg", "--qg", help="Do FS on qg-dataset",action="store_true")
parser.add_argument("--split", help="training split", type=int)
parser.add_argument("--feature", help="feature number", type=int)
parser.add_argument("--exp_name", help="unique name for the run", type=str)
#parser.add_argument("square",help="squares the input of the file", type=int,choices = [0,1,2,3,4])

args = parser.parse_args()

split = args.split
variable = args.variable


start=time.time()

save_dir = f'results/{args.exp_name}'




if args.tops:
	with open("/scratch/rd804/data/data/y_train.txt", "rb") as fp:
	    trainlabels = np.asarray(pickle.load(fp))
	with open("/scratch/rd804/data/data/y_test.txt", "rb") as fp:
	    testlabels = np.asarray(pickle.load(fp))
	with open("/scratch/rd804/data/data/y_val.txt", "rb") as fp:
	    vallabels = np.asarray(pickle.load(fp))

if args.qg:
	trainlabels = np.load(args.exp_name+'features/trainlabels.npy')
	testlabels = np.load(args.exp_name+'features/testlabels.npy')
	vallabels = np.load(args.exp_name+'features/vallabels.npy')	


ydisco = np.vstack((trainlabels.reshape(-1,1),vallabels.reshape(-1,1))).flatten()


batch_size=512
epochs=500

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

print(gpus)

num_classes=2

	
traindata = np.load(f'{save_dir}features/train_{args.exp_name}.npy')
testdata = np.load(f'{save_dir}features/test_{args.exp_name}.npy')
valdata = np.load(f'{save_dir}features/val_{args.exp_name}.npy')

print(traindata.shape)	
alldata = np.vstack((traindata,valdata))



scaler = RobustScaler()
scaler.fit(alldata)
alldata = scaler.transform(alldata)
valdata = scaler.transform(valdata)
testdata = scaler.transform(testdata)
traindata = scaler.transform(traindata)

print('....... training data ..... : ',traindata.shape)
print('....... all data ..... : ',alldata.shape)

ytrain = tf.keras.utils.to_categorical(trainlabels, num_classes)
yval = tf.keras.utils.to_categorical(vallabels, num_classes)
ytest = tf.keras.utils.to_categorical(testlabels, num_classes)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(traindata.shape[-1],)))
model.add(Dense(32, activation='relu'))
#model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

checkpoint_filepath_auc = '/scratch/rd804/training_variance_checks/model/model_'+str(args.exp_name)+'/checkpoint_'+str(m)+'_'+str(split)
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate =0.001))

model_checkpoint_callback_auc = tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_filepath_auc,
	save_weights_only=True,
	monitor='val_loss',
	mode='auto',
	save_best_only=True,min_delta=1e-10)    


history = model.fit(traindata, ytrain,
		batch_size=batch_size,
		epochs=epochs,
		verbose=0,
		validation_data=(valdata, yval),callbacks=[model_checkpoint_callback_auc])

model.load_weights(checkpoint_filepath_auc)
ypred=model.predict(testdata)[:,1]
t50,fpr50,_ = threshold_30(testlabels,ypred,0.3)
print('Background rejection is: ', 1/fpr50)

end=time.time()
print(end-start,'time taken')

r30 = np.asarray([1/fpr50])

if not os.path.exists('./performance/r30_variance_'+str(args.exp_name)):
	os.makedirs('./performance/r30_variance_'+str(args.exp_name))

if not os.path.exists('./performance/r30_variance_'+str(args.exp_name)+'/r30_'+str(m)):
	os.makedirs('./performance/r30_variance_'+str(args.exp_name)+'/r30_'+str(m))


np.save('./performance/r30_variance_'+str(args.exp_name)+'/r30_'+str(m)+'/r30_'+str(split)+'.npy',r30)


