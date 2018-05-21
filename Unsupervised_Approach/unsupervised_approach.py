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



def printRepartition(labels_pred, n_clusters, frac=0.05):

    nb = len(labels_pred)
    nbSp1PerLabel = [0 for i in range(n_clusters)]
    nbSp2PerLabel = [0 for i in range(n_clusters)]
    nbSp3PerLabel = [0 for i in range(n_clusters)]
    nbL1PerLabel = [0 for i in range(n_clusters)]
    nbL2PerLabel = [0 for i in range(n_clusters)]
    nbL3PerLabel = [0 for i in range(n_clusters)]
    nbL4PerLabel = [0 for i in range(n_clusters)]
    #ajouter une categorie pour les labels_pred -1
    for i in range(0, nb,int(1/frac)):
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

#### MAIN
path_file = open("mypath.txt", "r")
path = path_file.read()
print(path)
path_file.close()

os.chdir(path)

def createDatasets():
    for featuresType in ['MFCC', 'Speaker_trait']
        df_train,df_test = data.data(featuresType)

        df_train = df_train.fillna(0)#all NaN replaced by 0
        df_test = df_test.fillna(0)#all NaN replaced by 0

        # df_train = df_train.sample(n=5000, random_state=17)
        # df_test = df_test.sample(n=10000, random_state=17)

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

        np.savez('UL/215832_frames-'+featuresType+'-dataset.npz', data=dataset['data'], target=dataset['target'], language=dataset['language'])

        # dimRed = PCA(n_components=40)
        # X = dimRed.fit_transform(dataset['data'])


# ------------- Creating data ---------------------
def trainAndSave(n_clusters, featuresType, algorithm):
    print()
    print("--------------------------------------")
    print('Generating data for:')
    print("Number of clusters "+str(n_clusters))
    print("Features "+featuresType)
    print("Algorithm "+algorithm)

    dataset = np.load('UL/215832_frames-'+featuresType+'-dataset.npz')
    X = dataset['data']
    

    if (algorithm == 'KMeans'):
        labels_pred = KMeans(n_clusters=n_clusters, random_state=1).fit_predict(X)
    if (algorithm == 'GaussianMixture'):
        labels_pred = GaussianMixture(n_components=n_clusters, random_state=1).fit(X).predict(X)

    np.savez('UL/'+algorithm+'-'+featuresType+'-215832_frames-'+str(n_clusters)+'clusters.npz', 
        data=labels_pred)


# ------------- Analysing data ---------------------
def analyse(n_clusters, featuresType, algorithm):
    print()
    print("--------------------------------------")
    print('Analysing data for:')   
    print("Number of clusters "+str(n_clusters))
    print("Features "+featuresType)
    print("Algorithm "+algorithm)
    
    dataset = np.load('UL/215832_frames-'+featuresType+'-dataset.npz')
    labels_true = dataset['target']

    path = 'UL/'+algorithm+'-'+featuresType+'-215832_frames-'+str(n_clusters)+'clusters.npz'

    my_file = Path(path)
    if not(my_file.is_file()):
        trainAndSave(n_clusters, featuresType, algorithm)
    labels_pred = np.load(path)['data']
    print("rand score: "+str(metrics.adjusted_rand_score(labels_true, labels_pred)))   
    if (metrics.adjusted_rand_score(labels_true, labels_pred) > 0.01):
       #printRepartition(labels_pred, n_clusters=n_clusters)
        print("ACP = "+str(acp(labels_true, labels_pred, n_clusters)))



for alg in ['KMeans', 'GaussianMixture']:
    for feat in ["MFCC", "Speaker_trait"]:
        for m in [3, 6, 8, 10]:
            analyse(m, feat, alg)







# --------------------

# print("AgglomerativeClustering")
# labels_pred = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)
# np.savez('UL/AgglomerativeClustering-'+featuresType+'-215832_frames-'+str(n_clusters)+'clusters.npz', data=labels_pred)
# print("AgglomerativeClustering - DONE")

# print("AgglomerativeClustering")
# labels_pred = np.load('UL/AgglomerativeClustering-'+featuresType+'-215832_frames-'+str(n_clusters)+'clusters.npz')['data']
# print("rand score: "+str(metrics.adjusted_rand_score(labels_true, labels_pred)))   
# if (False and metrics.adjusted_rand_score(labels_true, labels_pred) > 0):
#     printRepartition(labels_pred)
