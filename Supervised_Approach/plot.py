#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:50:47 2018

@author: Younes
"""
from sklearn.svm import SVC
#from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

import numpy as np
import data, os, json
import copy

import itertools

import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler, Normalizer

from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration

from sklearn.mixture import GaussianMixture

from sklearn import neighbors

from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

path_file = open("mypath.txt", "r")
path = path_file.read()
print(path)
path_file.close()

os.chdir(path)



## Plot data with PCA to 2:

df_train,df_test = data.data('MFCC')
#df_train,df_test = data.data('Speaker_trait')

df_train = df_train.fillna(0)#all NaN replaced by 0
df_test = df_test.fillna(0)#all NaN replaced by 0

df_train = df_train.sample(n=5000, random_state=17)
df_test = df_test.sample(n=10000, random_state=17)

dataset = data.restructure_data(df_train)
dataset_test = data.restructure_data(df_test)

dataset['data'] = np.concatenate([dataset['data'].values.astype('float'), dataset_test['data'].values.astype('float')])
dataset['target'] = np.concatenate([dataset['target'].values.astype('str'),
                             dataset_test['target'].values.astype('str')])
dataset['language'] = np.concatenate([dataset['language'].values.astype('str'),
                             dataset_test['language'].values.astype('str')])
print("Scaling")
standard_scaler = StandardScaler()
dataset['data'] = standard_scaler.fit_transform(dataset['data'])


print("Dataset size "+str(len(dataset['data'])))

# reduce dimentionnality a bit if too high
# dimRed = PCA(n_components=50)
# dataset['data'] = dimRed.fit_transform(dataset['data'])
print("dimentionnality reduction")
dimRed = TSNE(n_components=2, verbose=1)
#dimRed = PCA(n_components=2)


dataset['data'] = dimRed.fit_transform(dataset['data'])

print("dimentionnality reduction done")


n = len(dataset['target'])
dataset['target_speaker_language'] = []
for i in range(n):
    dataset['target_speaker_language'] += [dataset['target'][i] + dataset['language'][i]]
    
# m = len(set(dataset['target_speaker_language']))
# indexType = list(set(dataset['target_speaker_language']))

m = len(set(dataset['target_speaker_language']))
indexType = list(set(dataset['target_speaker_language']))

separatedData = [[] for i in range(m)]

print(m)
print(indexType)

for i in range(n):
    #typeSpeaker = indexType.index(dataset['language'][i])
    typeSpeaker = indexType.index(dataset['target_speaker_language'][i])
    separatedData[typeSpeaker] += [dataset['data'][i]]


colors = ['b','g','r','c','m','y','k','orange']
for j in range(len(indexType)):
    x = [elem[0] for elem in separatedData[j]]
    y = [elem[1] for elem in separatedData[j]]
    plt.plot(x, y,color=colors[j], marker='+', label=indexType[j], linestyle='')
    plt.legend()
plt.show()

#np.savez('TSNE-Speaker_trait-15000_frames(PCA-50).npz', data=dataset)
#np.savez('TSNE-MFCC-15000_frames.npz', data=dataset)