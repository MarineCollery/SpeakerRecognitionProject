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
## Plot data with PCA to 2:

df, df_mfcc  = data.main();

df_train = df_mfcc.sample(frac=0.05, random_state=1)

dataset = data.restructure_data(df_train)

dataset['data'] = dataset['data'].values.astype('float')
dataset['target'] = dataset['target'].values.astype('str')
dataset['language'] = dataset['language'].values.astype('str')

standard_scaler = StandardScaler()
dataset['data'] = standard_scaler.fit_transform(dataset['data'])

dimRed = PCA(n_components=2)

dataset['data'] = dimRed.fit_transform(dataset['data'])

n = len(dataset['target'])
dataset['target_speaker_language'] = []
for i in range(n):
    dataset['target_speaker_language'] += [dataset['target'][i] + dataset['language'][i]]
    
# m = len(set(dataset['target_speaker_language']))
# indexType = list(set(dataset['target_speaker_language']))

m = len(set(dataset['language']))
indexType = list(set(dataset['language']))

separatedData = [[] for i in range(m)]

print(m)
print(indexType)

for i in range(n):
    #typeSpeaker = indexType.index(dataset['language'][i])
    typeSpeaker = indexType.index(dataset['target_speaker_language'][i])
    separatedData[typeSpeaker] += [dataset['data'][i]]


colors = ['b','g','r','c','m','y','k','w']
for j in range(len(indexType)):
    x = [elem[0] for elem in separatedData[j]]
    y = [elem[1] for elem in separatedData[j]]
    plt.plot(x, y,color=colors[j], marker='+', label=indexType[j], linestyle='')
    plt.legend()
plt.show()