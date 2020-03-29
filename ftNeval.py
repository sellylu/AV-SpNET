#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 19:12:13 2018

@author: cclu
"""
target = 'act'
cuda   = '1'
condition = 'act'
verb   = 1
dataset= 'IEMOCAP'
import sys
if len(sys.argv)>=2:
	condition = sys.argv[1]
if len(sys.argv)>=3:
	dataset = sys.argv[2]
if len(sys.argv)>=4:
    target = sys.argv[3]
if len(sys.argv)>=5:
	cuda = sys.argv[4]
if len(sys.argv)>=6:
	verb = int(sys.argv[5])

print('**********\n'+ \
	   'use "'+condition+'avspnet" initial\nclassify "'+target+'" of "'+dataset+'" on gpu'+cuda+'\n'+ \
	   '**********')


data_dir    = 'data/' + dataset + '/'
model_path  = './net/cnn_encoder.h5'
weight_path = './net/avspnet_'+condition+'.pkl'

import os
import numpy as np
np.set_printoptions(precision=2)
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = cuda
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import matplotlib.pyplot as plt
import itertools

'''plotting cm'''
def plotCM(cm, classes,
           normalize=False,
           title='Confusion matrix',
           cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def calAccuracy(true, pred, class_names, title='', draw=True):
	from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
	acc = accuracy_score(true, pred)
	uar = recall_score(true, pred, average='macro')
	cm  = confusion_matrix(true, pred, range(CLASS))
	print("acc: {}\tuar: {}\ncm: \n{}".format(acc, uar, cm))
	print(title)
	if draw:
		plotCM(cm, classes=class_names, title='Confusion matrix('+title+')')
	return (acc, uar, cm)

#%%
import pickle
with open(data_dir+'sxx_16_log_800_80.pkl', 'rb') as f:
	train_data = pickle.load(f, encoding='latin1')

#%%
raw = pd.read_csv(data_dir+'label.csv', index_col='Name_String')
#raw = raw[raw['Emotion'] != 'xxx']
#raw = raw[raw['Emotion'] != 'oth']

class_names = ['Deactivation', 'Activation']
if target == 'cat4':
	selection = ~raw[target].isnull()
	raw = raw[selection]
	class_names = ['hap', 'ang', 'sad', 'neu']
CLASS = len(class_names)

X_train = list()
Y_train = list()
Y_index = list()
speaker = list()
for index, row in raw.iterrows():
	try:
		x = train_data[index]
		X_train.extend(x)
		if target != 'cat4':
			Y_train.extend([row[target+'_spk']]*len(x))
		else:
			Y_train.extend([row[target]]*len(x))
		Y_index.extend([index]*len(x))
		speaker.extend([row['speaker']]*len(x))
	except:
		continue
del raw_data, train_data

X_train = np.array(X_train)
X_train = np.reshape(X_train, X_train.shape+(1,))

from keras.utils import np_utils
Y_train = np.array(Y_train, dtype=int)
Y_train_hot = np_utils.to_categorical(Y_train, CLASS)


Y_index, speaker = map(np.array, (Y_index, speaker))

#%%
'''load pre-trained model'''
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Flatten
from keras import backend as K

FRAC = 5
SPK = len(np.unique(speaker))//2
GP  = SPK//FRAC

for train_por in range(GP,SPK-GP+1,GP):
	all_uar=[]; all_uar_voted=[]
	Y_test_cv = []; pred_cv = []
	for select in range(1,FRAC+1):
		speaker_cv = speaker-GP*select*2
		speaker_cv[speaker_cv<=0] += (SPK*2)
		train_i = np.where(speaker_cv<=train_por*2)
		test_i  = np.where(speaker_cv> (SPK-GP)*2)

#		print('train portion:',train_por,'\tfold:', select)

		model = load_model(model_path)

		model.add(Flatten())
		model.add(Dense(1024, activation='relu', name='fc_add1'))
		model.add(Dense( 512, activation='relu', name='fc_add2'))
		model.add(Dense( 128, activation='relu', name='fc_add3'))
		model.add(Dropout(0.5, name='drop_add1'))
		model.add(Dense(CLASS, activation='softmax', name='output'))

		with open(weight_path, 'rb') as f:
			weights = pickle.load(f)
		for i,w in enumerate(weights):
			if i < 10:
				model.layers[i].trainable = False
				model.layers[i].set_weights(w)

		'''setting optimizer'''
		from keras.optimizers import SGD
		lr = 0.0001
		if dataset=='NNIME':
			lr = 0.0001
		decay = 1e-4
		mom = 0.8
		opt = SGD(lr=lr, decay=decay, momentum=mom, nesterov=True)
		model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

		#%
		''' training'''
		batch_size = 32
		epoch = 15

		from keras.callbacks import EarlyStopping

		earlyStopping = EarlyStopping(monitor='val_loss', patience=5)

		fit_log = model.fit(	X_train[train_i], Y_train_hot[train_i],
									batch_size=batch_size,
									epochs=epoch,
									verbose=verb,
									shuffle=True,
									validation_split=0.1,
									class_weight='balanced',
									callbacks=[	earlyStopping])

		#%
		'''prediction analysis'''
		Y_test = Y_train[test_i]
		Y_test_ind = Y_index[test_i]
		pred = model.predict_classes(X_train[test_i], batch_size, verbose=0)

		from scipy.stats import mode
		pred_voted = list()
		Y_test_voted = list()
		for a in np.unique(Y_test_ind):
			ind = np.where(Y_test_ind == a)
		#	print(mode(pred[ind]), Y_test[ind][0])
			pred_voted.append(mode(pred[ind])[0][0])
			Y_test_voted.append(Y_test[ind][0])

		Y_test_cv.extend(Y_test_voted)
		pred_cv.extend(pred_voted)

		_,uar_voted,_ = calAccuracy(Y_test_voted, pred_voted, class_names, title='voted', draw=False)

		all_uar_voted.append(uar_voted)
		del model
		K.clear_session()
	print(train_por,'\t',','.join(map(str,all_uar_voted)))

	_, uar, _ = calAccuracy(Y_test_cv, pred_cv, class_names, title='voted', draw=False)

