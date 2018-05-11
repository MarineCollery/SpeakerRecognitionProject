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





# Read Data ,Pre Processing, Dimensionality Reduction from data.py


# Classification Model


# generation confusion Matrix and ROC curve if possible

def trainAndEvaluate(dataset, dataset_test, n_components=40, dimReduction='PCA', classifier="SVC"):

    ## Pre processing
    standard_scaler = StandardScaler()
    dataset['data'] = standard_scaler.fit_transform(dataset['data'])
    dataset_test['data'] = standard_scaler.transform(dataset_test['data'])

    ## Dimensionality reduction
    if (dimReduction == 'PCA'):
        dimRed = PCA(n_components=n_components)
    elif (dimReduction == 'FA'):
        dimRed = FeatureAgglomeration(n_clusters=n_components)

    dataset['data'] = dimRed.fit_transform(dataset['data'])
    dataset_test['data'] = dimRed.transform(dataset_test['data'])

    ## Classifier initialisation
    if (classifier == 'SVC'):
        clf = SVC(C=1, class_weight='balanced', verbose=0, probability=True)
    elif (classifier == 'kNN'):
        clf = neighbors.KNeighborsClassifier(n_neighbors=10)
    elif (classifier == 'tree'):
        clf = DecisionTreeClassifier(class_weight ='balanced', random_state=1)


    print("Training...")
    clf.fit(dataset['data'], dataset['target'])
    print("Prediciting...")
    predicted = clf.predict(dataset_test['data'])
    accuracy = np.mean(predicted == dataset_test['target'])

    return accuracy


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
    df_test = df_mfcc.drop(df_train.index)
    
    df_test = df_test.sample(frac=0.02, random_state=1)
    
    dataset = data.restructure_data(df_train)
    dataset_test = data.restructure_data(df_test)
    
    dataset_original = copy.deepcopy(dataset['data'])
    dataset_test_original = copy.deepcopy(dataset_test['data'])

    # accuracies = []
    # for k in range(1, 41,4):
    #     print('k='+str(k))

    #     dataset['data'] = dataset_original
    #     dataset_test['data'] = dataset_test_original

    #     dataset['data'] = dataset['data'].values.astype('float')
    #     dataset_test['data'] =  dataset_test['data'].values.astype('float')

    #     accuracies += [trainAndEvaluate(dataset, dataset_test, k)]
    # print (accuracies)
    #[0.36912254720473897, 0.5127730470196223, 0.6016290262865606, 0.6373565346168086, 0.6782673084042947, 0.6965938541281007, 0.7252869307663828, 0.7597186227323214, 0.7728619029988893, 0.8028507960014809]
    
    # accuracies = []
    # for k in range(37, 42):
    #     print('k='+str(k))

    #     dataset['data'] = dataset_original
    #     dataset_test['data'] = dataset_test_original

    #     dataset['data'] = dataset['data'].values.astype('float')
    #     dataset_test['data'] =  dataset_test['data'].values.astype('float')

    #     accuracies += [trainAndEvaluate(dataset, dataset_test, k)]
    # print (accuracies)
    # [0.8028507960014809, 0.8008145131432802, 0.8182154757497223, 0.8343206219918549, 0.8335801554979637]
    
    # accuracies = []
    # for k in range(1, 41,4):
    #     print('k='+str(k))

    #     dataset['data'] = dataset_original
    #     dataset_test['data'] = dataset_test_original

    #     dataset['data'] = dataset['data'].values.astype('float')
    #     dataset_test['data'] =  dataset_test['data'].values.astype('float')

    #     accuracies += [trainAndEvaluate(dataset, dataset_test, k, 'FA')]
    # print (accuracies)
    #[0.3674564975934839, 0.4833395038874491, 0.5664568678267309, 0.5988522769344687, 0.6436504998148834, 0.6677156608663458, 0.7191780821917808, 0.7497223250647909, 0.7848944835246205, 0.8052573121066272]

    # accuracies = []
    # for k in range(37, 42):
    #     print('k='+str(k))

    #     dataset['data'] = dataset_original
    #     dataset_test['data'] = dataset_test_original

    #     dataset['data'] = dataset['data'].values.astype('float')
    #     dataset_test['data'] =  dataset_test['data'].values.astype('float')

    #     accuracies += [trainAndEvaluate(dataset, dataset_test, k, 'FA')]
    # print (accuracies)
    # [0.8052573121066272, 0.8283968900407257, 0.8328396890040726, 0.8341355053683821, 0.8335801554979637]
    
    # on this set the best outcome is with 40 features

    ## SVM
    dataset['data'] = dataset_original
    dataset_test['data'] = dataset_test_original

    dataset['data'] = dataset['data'].values.astype('float')
    dataset_test['data'] =  dataset_test['data'].values.astype('float')

    print(trainAndEvaluate(dataset, dataset_test, classifier='SVC'))
    # 0.8343206219918549
    ## kNN
    dataset['data'] = dataset_original
    dataset_test['data'] = dataset_test_original

    dataset['data'] = dataset['data'].values.astype('float')
    dataset_test['data'] =  dataset_test['data'].values.astype('float')

    print(trainAndEvaluate(dataset, dataset_test, classifier='kNN'))
    # 0.7051092188078489

    ## kNN
    dataset['data'] = dataset_original
    dataset_test['data'] = dataset_test_original

    dataset['data'] = dataset['data'].values.astype('float')
    dataset_test['data'] =  dataset_test['data'].values.astype('float')

    print(trainAndEvaluate(dataset, dataset_test, classifier='tree'))
    # 0.5194372454646428


    # Read Data
    
    
    # Pre Processing
    
    
    # Dimensionality Reduction



if __name__ == "__main__":
    main()

