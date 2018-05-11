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
    df_test = df_mfcc.drop(df_train.index)
    
    df_test = df_test.sample(frac=0.02, random_state=1)
    
    dataset = data.restructure_data(df_train)
    dataset_test = data.restructure_data(df_test)
    
    ## Pre processing
    scaler = StandardScaler()
    dataset['data'] = scaler.fit_transform(dataset['data'].values.astype('float'))
    dataset_test['data'] = scaler.transform(dataset_test['data'].values.astype('float'))


    clf = SVC(C=1, class_weight='balanced', verbose=1, probability=True)
    print("Training...")
    clf.fit(dataset['data'], dataset['target'])
    print("Prediciting...")
    predicted = clf.predict(dataset_test['data'])
    print("\naccuracy : "+str(np.mean(predicted == dataset_test['target'])))
    
    # Read Data
    
    
    # Pre Processing
    
    
    # Dimensionality Reduction



if __name__ == "__main__":
    main()

