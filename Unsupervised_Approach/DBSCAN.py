#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:50:47 2018

@author: Marine
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


import itertools

import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration

from sklearn.cluster import DBSCAN, SpectralClustering, KMeans, AgglomerativeClustering

from sklearn.mixture import GaussianMixture

from sklearn import metrics

from pathlib import Path

onlyFrench = True

path_file = open("mypath.txt", "r")
path = path_file.read()
print(path)
path_file.close()

os.chdir(path)


def printRepartition(labels_pred, n_clusters):

    nb = len(labels_pred)
    nbSp1PerLabel = [0 for i in range(n_clusters)]
    nbSp2PerLabel = [0 for i in range(n_clusters)]
    nbSp3PerLabel = [0 for i in range(n_clusters)]
    nbL1PerLabel = [0 for i in range(n_clusters)]
    nbL2PerLabel = [0 for i in range(n_clusters)]
    nbL3PerLabel = [0 for i in range(n_clusters)]
    nbL4PerLabel = [0 for i in range(n_clusters)]
    #ajouter une categorie pour les labels_pred -1
    for i in range(0, nb):
        currentLabel = labels_pred[i]
        currentSpeaker = dataset['target'][i]
        currentLanguage = dataset['language'][i]

        if(currentSpeaker == "Sp1"):
           nbSp1PerLabel[currentLabel] += 1

        elif(currentSpeaker == "Sp2"):
           nbSp2PerLabel[currentLabel] += 1

        elif(currentSpeaker == "Sp3"):
           nbSp3PerLabel[currentLabel] += 1

        if (currentLanguage == "L1"):
           nbL1PerLabel[currentLabel] += 1

        elif (currentLanguage == "L2"):
           nbL2PerLabel[currentLabel] += 1

        elif (currentLanguage == "L3"):
           nbL3PerLabel[currentLabel] += 1

        elif (currentLanguage == "L4"):
           nbL4PerLabel[currentLabel] += 1

    print("Repartition of Sp1: " + str(nbSp1PerLabel));
    print("Repartition of Sp2: " + str(nbSp2PerLabel));
    print("Repartition of Sp3: " + str(nbSp3PerLabel));
    print("Repartition of L1: " + str(nbL1PerLabel));
    print("Repartition of L2: " + str(nbL2PerLabel));
    print("Repartition of L3: " + str(nbL3PerLabel));
    print("Repartition of L4: " + str(nbL4PerLabel));

def acp(labels_true, labels_pred, n_clusters):
    """
    Computes average cluster purity
    """
    speakers = ["Sp1", "Sp2", "Sp3"]
    assert(len(labels_pred)==len(labels_true))
    # total number of frames
    N = len(labels_pred)
    # number of speakers
    Ns = 3
    # number of clusters
    Nc = n_clusters
    # computing the values of the n matrix, nij = number of speaker j in cluster i
    n = [[0 for j in range(Ns)] for i in range(Nc)]
    for k in range(N):
        i = labels_pred[k]
        j = speakers.index(labels_true[k])

        n[i][j] += 1

    nn = [0 for i in range(Nc)]

    for i in range(Nc):
        for j in range(Ns):
            nn[i] += n[i][j]

    pi = [0 for i in range(Nc)]
    for i in range(Nc):
        for j in range(Ns):
            pi[i] += n[i][j]**2/nn[i]**2
    acp = 0
    for i in range(Nc):
        acp += pi[i]*nn[i]
    acp = acp/N
    return acp

def acp_language(labels_true, labels_pred, n_clusters):
    """
    Computes average cluster purity
    """
    speakers = ["L1", "L2", "L3", "L4"]
    assert(len(labels_pred)==len(labels_true))
    # total number of frames
    N = len(labels_pred)
    # number of speakers
    Ns = 4
    # number of clusters 
    Nc = n_clusters
    # computing the values of the n matrix, nij = number of speaker j in cluster i
    n = [[0 for j in range(Ns)] for i in range(Nc)]
    for k in range(N):
        i = labels_pred[k]
        j = speakers.index(labels_true[k])
        
        n[i][j] += 1

    nn = [0 for i in range(Nc)]

    for i in range(Nc):
        for j in range(Ns):
            nn[i] += n[i][j]
    
    pi = [0 for i in range(Nc)]
    for i in range(Nc):
        for j in range(Ns):
            pi[i] += n[i][j]**2/nn[i]**2
    acp = 0
    for i in range(Nc):
        acp += pi[i]*nn[i]
    acp = acp/N
    return acp



#DBSCAN
def dbscanAlgorithme(eps, min_samples):
    featuresType = 'MFCC' # 'MFCC' or 'Speaker_trait'
    df_train,df_test = data.data(featuresType)
    #Delete unwanted colums
    if 'frameIndex' in df_train.columns.values.tolist():
    #Database MFCC
        columns = ['Unnamed: 0', 'frameIndex', 'frameTime']
        df_train.drop(columns, inplace=True, axis=1)
        df_test.drop(columns, inplace=True, axis=1)
    elif 'Unnamed: 0.1' in df_train.columns.values.tolist():
        #Database Speaker trait
        columns = ['Unnamed: 0', 'Unnamed: 0.1']
        df_train.drop(columns, inplace=True, axis=1)
        df_test.drop(columns, inplace=True, axis=1)
    df_train = df_train.fillna(0)#all NaN replaced by 0
    df_test = df_test.fillna(0)#all NaN replaced by 0

    # df_train = df_train.sample(n=5000, random_state=17)
    df_test = df_test.sample(n=50000, random_state=17)

    dataset = data.restructure_data(df_train)
    dataset_test = data.restructure_data(df_test)
    dataset['data'] = np.concatenate([dataset['data'].values.astype('float'), dataset_test['data'].values.astype('float')])
    dataset['target'] = np.concatenate([dataset['target'].values.astype('str'),
                                 dataset_test['target'].values.astype('str')])
    dataset['language'] = np.concatenate([dataset['language'].values.astype('str'),
                                 dataset_test['language'].values.astype('str')])
    
    X = dataset['data']
    labels_true = dataset['target']
    labels_language = dataset['language']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    dimRed = DBSCAN(eps , min_samples , metric= 'euclidean', metric_params = None, algorithm = 'auto', leaf_size= 20, n_jobs =  -1);
    db = dimRed.fit(X)
    labels = db.labels_
    n_clusters_ = len(set(labels))
    np.savez('UL/DBSCAN.npz', labels=labels, n_clusters = n_clusters_)
    print("ARI = "+str(metrics.adjusted_rand_score(labels_true, labels)))
    print("ACP = "+str(acp(labels_true,labels, n_clusters_)))
    print("ACP_language = "+str(acp_language(labels_language,labels, n_clusters_)))

    #printRepartition(labels, n_clusters_)


dbscanAlgorithme(3.5535, 17)