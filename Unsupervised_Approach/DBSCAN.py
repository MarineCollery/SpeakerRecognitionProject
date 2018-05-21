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



# Read Data ,Pre Processing, Dimensionality Reduction from data.py


# Classification Model


# generation confusion Matrix and ROC curve if possible

def main():
    df, df_mfcc  = data.main();
    
    
#    #TO DO : ADAPT FEATUERS
#    
#    df_train = df.sample(frac=0.8, random_state=1)
#    df_test = df.drop(df_train.index)
#    
#    dataset = data.restructure_data(df_train)
#    dataset_test = data.restructure_data(df_test)
#    
#    
#
#    clf = SVC(C=1, class_weight='balanced', verbose=0, probability=True)
#    clf.fit(dataset['data'].values.astype('float'), dataset['target'])
#    
#    predicted = clf.predict(dataset_test['data'].values.astype('float'))
#    print("\naccuracy : "+str(np.mean(predicted == dataset_test['target'])))
    
    df_train = df_mfcc.sample(frac=0.02, random_state=1)
    
    dataset = data.restructure_data(df_train)

    dataset['data'] = dataset['data'].values.astype('float')
    X = dataset['data']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    labels_true = dataset['target']

    dimRed = PCA(n_components=40)
    X = dimRed.fit_transform(dataset['data'])

    labels_pred = SpectralClustering(n_clusters=3, random_state=1).fit_predict(X)
    print("rand score: "+str(metrics.adjusted_rand_score(labels_true, labels_pred)))
    # rand score: 0.00016338249749170178
   
    labels_pred = KMeans(n_clusters=3, random_state=1).fit_predict(X)
    print("rand score: "+str(metrics.adjusted_rand_score(labels_true, labels_pred)))
    # rand score: 0.02133033310820022

    labels_pred = AgglomerativeClustering(n_clusters=3).fit_predict(X)
    print("rand score: "+str(metrics.adjusted_rand_score(labels_true, labels_pred)))
    # rand score: 0.016111491678891945

    labels_pred = GaussianMixture(n_components=3, random_state=1).fit(X).predict(X)
    print("rand score: "+str(metrics.adjusted_rand_score(labels_true, labels_pred)))
    # rand score: 0.02532967351339382


    # DBSCAN
#    dt_DBSCAN = StandardScaler().fit_transform(dataset['data'])
#    dimRed = DBSCAN(eps = 3.5535, min_samples = 17, metric= 'euclidean', metric_params = None, algorithm = 'auto', leaf_size= 20, n_jobs =  -1);
#    db = dimRed.fit(dt_DBSCAN)
#    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
#    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
#    print('Estimated number of clusters: %d' % n_clusters_)
    #        print(len(labels))
    #        print(labels[:100])
    #        print(len(dataset['target']))

#    print(metrics.adjusted_rand_score(dataset['language'].values.astype('str'), labels))

#    nb = len(labels)
#    nbSp1PerLabel = [0 for i in range(n_clusters_+1)]
#    nbSp2PerLabel = [0 for i in range(n_clusters_+1)]
#    nbSp3PerLabel = [0 for i in range(n_clusters_+1)]
#    nbL1PerLabel = [0 for i in range(n_clusters_+1)]
#    nbL2PerLabel = [0 for i in range(n_clusters_+1)]
#    nbL3PerLabel = [0 for i in range(n_clusters_+1)]
#    nbL4PerLabel = [0 for i in range(n_clusters_+1)]
    # ajouter une categorie pour les labels -1
#    for i in range(nb):
#        currentLabel = labels[i]
#        currentSpeaker = dataset['target'].values.astype('str')[i]
#        currentLanguage = dataset['language'].values.astype('str')[i]
#        
#        if(currentSpeaker == "Sp1"):
#            nbSp1PerLabel[currentLabel] += 1
            
#        elif(currentSpeaker == "Sp2"):
#            nbSp2PerLabel[currentLabel] += 1
            
#        elif(currentSpeaker == "Sp3"):
#            nbSp3PerLabel[currentLabel] += 1

#        if (currentLanguage == "L1"):
#            nbL1PerLabel[currentLabel] += 1

#        elif (currentLanguage == "L2"):
#            nbL2PerLabel[currentLabel] += 1
                
#        elif (currentLanguage == "L3"):
#            nbL3PerLabel[currentLabel] += 1
                
#        elif (currentLanguage == "L4"):
#            nbL4PerLabel[currentLabel] += 1
                
#        print("Repartition of Sp1: " + str(nbSp1PerLabel));
#        print("Repartition of Sp2: " + str(nbSp2PerLabel));
#        print("Repartition of Sp3: " + str(nbSp3PerLabel));
#        print("Repartition of L1: " + str(nbL1PerLabel));
#        print("Repartition of L2: " + str(nbL2PerLabel));
#        print("Repartition of L3: " + str(nbL3PerLabel));
#        print("Repartition of L4: " + str(nbL4PerLabel));


#        for (i in range (len(dataset['target']))):
#            if ((dataset['target'].values.astype('str') == dataset['target_names'][0])):
                 

             
             
    # Read Data
    
    
    # Pre Processing
    
    
    # Dimensionality Reduction



if __name__ == "__main__":
    main()

