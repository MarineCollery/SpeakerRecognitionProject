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
from sklearn.metrics import classification_report
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

import pandas as pd




# Read Data ,Pre Processing, Dimensionality Reduction from data.py


# Classification Model


# generation confusion Matrix and ROC curve if possible

def trainAndEvaluate(dataset, dataset_test, n_components=40, dimReduction='PCA', classifier="SVC", preproc_speaker=False):

    ## Pre processing
    if(preproc_speaker):
        print("preprocessing has been computed previously")

    else:
        standard_scaler = StandardScaler()
        standard_scaler.fit(dataset['data'])
        dataset['data'] = standard_scaler.fit_transform(dataset['data'], standard_scaler.get_params())
        dataset_test['data'] = standard_scaler.transform(dataset_test['data'], standard_scaler.get_params())

#    ## Dimensionality reduction
#    
#    
#    if (dimReduction == 'LDA'):
#        dimRed = LinearDiscriminantAnalysis(n_components=n_components)
#        dataset['data'] = dimRed.fit(dataset['data'], dataset['target']).transform(dataset['data'])
#        dataset_test['data'] = dimRed.transform(dataset_test['data'])
#    else:
#        if (dimReduction == 'PCA'):
#            dimRed = PCA(n_components=n_components)
#        elif (dimReduction == 'FA'):
#            dimRed = FeatureAgglomeration(n_clusters=n_components)
#        dataset['data'] = dimRed.fit_transform(dataset['data'])
#        dataset_test['data'] = dimRed.transform(dataset_test['data'])

    ## Classifier initialisation
    if (classifier == 'SVC'):
        clf = SVC(C=1, class_weight='balanced', verbose=1, probability=True)
    elif (classifier == 'kNN'):
        clf = neighbors.KNeighborsClassifier(n_neighbors=10)
    elif (classifier == 'tree'):
        clf = DecisionTreeClassifier(class_weight ='balanced', random_state=1)

    
    print("Training...")
    clf.fit(dataset['data'], dataset['target'])
    print("Prediciting...")
    predicted = clf.predict(dataset_test['data'])
    
    report = classification_report(dataset_test['target'], predicted, target_names=dataset_test['target_names'])
    
    accuracy = np.mean(predicted == dataset_test['target'])
    cnf_matrix = confusion_matrix(dataset_test['target'], predicted)

    return accuracy, cnf_matrix, report

def standardization_speaker(df_train, df_test):
    
    standard_scaler = StandardScaler()
    
    # SP1
    df_train1 = df_train[(df_train.target_names=="Sp1")]
    features = df_train1.loc[:, df_train1.columns != 'name']
    features = features.loc[:, features.columns != 'target_names']
    features = features.loc[:, features.columns != 'language']
    
    column_names = features.columns.values.tolist()
    standard_scaler.fit(features)
    
    df_train1 = standard_scaler.fit_transform(features, standard_scaler.get_params())
    
    df_test1 = df_test[(df_test.target_names=="Sp1")]
    features = df_test1.loc[:, df_test1.columns != 'name']
    features = features.loc[:, features.columns != 'target_names']
    features = features.loc[:, features.columns != 'language']
    
    df_test1 = standard_scaler.transform(features, standard_scaler.get_params())
    
    
    # SP2
    df_train2 = df_train[(df_train.target_names=="Sp2")]
    features = df_train2.loc[:, df_train2.columns != 'name']
    features = features.loc[:, features.columns != 'target_names']
    features = features.loc[:, features.columns != 'language']
    
    standard_scaler.fit(features)
    df_train2 = standard_scaler.fit_transform(features, standard_scaler.get_params())
    
    df_test2 = df_test[(df_test.target_names=="Sp2")]
    features = df_test2.loc[:, df_test2.columns != 'name']
    features = features.loc[:, features.columns != 'target_names']
    features = features.loc[:, features.columns != 'language']
    
    df_test2 = standard_scaler.transform(features, standard_scaler.get_params())
    
    # SP3
    df_train3 = df_train[(df_train.target_names=="Sp3")]
    features = df_train3.loc[:, df_train3.columns != 'name']
    features = features.loc[:, features.columns != 'target_names']
    features = features.loc[:, features.columns != 'language']
    
    standard_scaler.fit(features)
    df_train3 = standard_scaler.fit_transform(features, standard_scaler.get_params())
    
    df_test3 = df_test[(df_test.target_names=="Sp3")]
    features = df_test3.loc[:, df_test3.columns != 'name']
    features = features.loc[:, features.columns != 'target_names']
    features = features.loc[:, features.columns != 'language']
    
    df_test3 = standard_scaler.transform(features, standard_scaler.get_params())
    
    #To dataframe type
    df_train1 = pd.DataFrame(data = df_train1, columns = column_names)
    df_train2 = pd.DataFrame(data = df_train2, columns = column_names)
    df_train3 = pd.DataFrame(data = df_train3, columns = column_names)
    
    df_test1 = pd.DataFrame(data = df_test1, columns = column_names)
    df_test2 = pd.DataFrame(data = df_test2, columns = column_names)
    df_test3 = pd.DataFrame(data = df_test3, columns = column_names)
     
    #concat
    df_train_fin = pd.concat([df_train1, df_train2, df_train3])
    df_test_fin = pd.concat([df_test1, df_test2, df_test3])
    
    #Reindex
    df_train_fin.index = range(df_train_fin.shape[0])
    df_test_fin.index = range(df_test_fin.shape[0])
    
    #concat with final columns
    df_train_fin = pd.concat([df_train_fin, df_train.loc[:, df_train.columns == 'language'], df_train.loc[:, df_train.columns == 'name'],df_train.loc[:, df_train.columns == 'target_names']], axis = 1)
    df_test_fin = pd.concat([df_test_fin, df_test.loc[:, df_test.columns == 'language'], df_test.loc[:, df_test.columns == 'name'],df_test.loc[:, df_test.columns == 'target_names']], axis = 1)
    
    #randomize
    df_train=df_train_fin.sample(frac=1,random_state=1)
    df_test=df_test_fin.sample(frac=1,random_state=1)
    
    
    return df_train, df_test


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")
#        plt.text(j, i, format(' '),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_un_and_normalized_cnf(cnf_matrix, classes, name):
    np.set_printoptions(precision=2)
   
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix,classes=classes, title='Confusion matrix, without normalization')
    plt.savefig(name+'.png')


    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, normalize=True, classes=classes, title='Normalized confusion matrix')
    plt.savefig(name+'_normalized.png')
#    plt.show()
    
def generate_plots(dataset_name, one_lang=False, language="L2", 
                   classifier="SVC", n_components=40, dimReduction='PCA', 
                   pre_speaker=False, nbr_elements=10000):
    
    
    if one_lang:
        df_train, df_test = data.data_ONE_LANG(dataset_name, language, nbr_elements) 
        name = dataset_name+'_'+language+'_'+'ONLY'+'_'+str(nbr_elements)+'_'+classifier+'_'+dimReduction+'_'+str(n_components)+'_pre_speaker_'+str(pre_speaker)
    else:
        df_train, df_test = data.data(dataset_name, language, all_train=False, nbr_elements=nbr_elements)
        name = dataset_name+'_'+language+'_'+str(nbr_elements)+'_train_'+'LEFT_FOR_TEST'+'_'+classifier+'_'+dimReduction+'_'+str(n_components)+'_pre_speaker_'+str(pre_speaker)
    
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
    
    if pre_speaker:
        #normalize and randomize
        df_train, df_test = standardization_speaker(df_train, df_test)
    else:
        #randomize
        df_train=df_train.sample(frac=1,random_state=1)
        df_test=df_test.sample(frac=1,random_state=1)
        
    #Restructure for the next steps (training)
    dataset = data.restructure_data(df_train)
    dataset_test = data.restructure_data(df_test)
    
#    if 'MFCC' in dataset_name:
#        #plot boxplot
#        mfcc = [str(i) for i in range(1,14)]
#        mfcc_de = [str(i)+'_de' for i in range(1,14)]
#        mfcc_de_de = [str(i)+'_de_de' for i in range(1,14)]
#        dataset['data'].columns = mfcc+mfcc_de+mfcc_de_de
#    #    plt.figure(figsize=(12,8))
#    #    bp = dataset['data'].boxplot(rot=45)
#    #    plt.savefig('MFCC'+'_box_plot.png')
#        
#        df = dataset['data'].loc[:,mfcc_de_de]
#        df['Speaker'] = pd.Series(dataset['target'])
#        plt.figure(figsize=(14,8))
#        bp = df.boxplot(by='Speaker',figsize=(12,8))
#        plt.savefig('MFCC_de_de'+'_box_plot_per_Speaker.png')
#        
#    else:
##        plt.figure(figsize=(15,15))
##        bp = dataset['data'].boxplot(rot=45)
##        plt.savefig('Speaker_trait'+'_box_plot.png')
#        print(len(dataset['data'].columns.values.tolist()))
#        columns = ['pcm_fftMag_spectralVariance_sma', 'pcm_fftMag_spectralVariance_sma.1', 'pcm_fftMag_spectralVariance_sma_de', 'pcm_fftMag_spectralVariance_sma_de.1']
#        df = dataset['data'].loc[:,columns]
#        df['Speaker'] = pd.Series(dataset['target'])
#        plt.figure(figsize=(14,8))
#        bp = df.boxplot(by='Speaker',figsize=(12,8))
#        plt.savefig('Speaker_trait_spectral_variance'+'_box_plot_per_Speaker.png')
    
    acc, cnf_matrix, report = trainAndEvaluate(dataset, dataset_test, n_components, dimReduction, classifier, pre_speaker)
    f = open(name+'.txt','w')
    f.write(report)
    f.write("\n Accuracy:%0.6f" % acc)
    f.close()
    print(acc)
    
    plot_un_and_normalized_cnf(cnf_matrix, dataset['target_names'], name)
    
    
#    return acc
    
    

def main():

    #Adding Feature path
    
    path_file = open("mypath.txt", "r")
    path = path_file.read()
    print(path)
    path_file.close()
    
    os.chdir(path)
    
    dataset_name ='Speaker_trait'
    dataset_name_MFCC ='MFCC'
    language="L2"
    frac_train=0.2
    classifier="SVC"
    
    #dimensionality reduction 
    n_components=40
    dimReduction='PCA'
    
    # Standardization on the speaker level
    pre_speaker = False
    
#    generate_plots(dataset_name_MFCC, one_lang=False, language="L2", 
#                   classifier="SVC", n_components=30, dimReduction='PCA', 
#                   pre_speaker=False, nbr_elements=10000)
    
    #EVOLUTION ofAcc with PCA
#    acc =[]
#    acc2 =[]
#    for i in range (9,39,3):
#        print("Generating %d" % i)
#        acc.append(generate_plots(dataset_name, one_lang=False, language="L2", 
#                   classifier="SVC", n_components=i, dimReduction='PCA', 
#                   pre_speaker=True, nbr_elements=10000))
#        acc2.append(generate_plots(dataset_name, one_lang=False, language="L2", 
#                   classifier="SVC", n_components=i, dimReduction='PCA', 
#                   pre_speaker=False, nbr_elements=10000))
#        
#    plt.plot(range (9,39,3),acc,'bo', label='std per speaker')
#    plt.plot(range (9,39,3),acc2,'bo', label='std on all data')
#    plt.legend()
#    plt.title("Accuracy evolution with nbr of components of PCA dimensionality reduction")
#    plt.xlabel("Accuracy")
#    plt.ylabel("nbr component")
#    plt.savefig(dataset_name+'_Evolution_Acc_PCA.png')
#    plt.show()
    
    
#    generate_plots(dataset_name, one_lang=False, language="L2", 
#                   classifier="SVC", n_components=2, dimReduction='PCA', 
#                   pre_speaker=False, nbr_elements=10000)
#    generate_plots(dataset_name_MFCC, one_lang=False, language="L2", 
#                   classifier="SVC", n_components=2, dimReduction='PCA', 
#                   pre_speaker=False, nbr_elements=10000)


#    #Generate plots for FR train OTHER test
#    acc_mfcc_true = np.zeros(3)
#    acc_mfcc_false = np.zeros(3)
#    acc_ST_true = np.zeros(3)
#    acc_ST_false = np.zeros(3)
#    
#    acc_mfcc_true[0] = generate_plots(dataset_name_MFCC, one_lang=False, language="L2", 
#                 classifier="SVC", n_components='all', dimReduction='PCA', 
#                 pre_speaker=True, nbr_elements=10000)
#    acc_mfcc_false[0] = generate_plots(dataset_name_MFCC, one_lang=False, language="L2", 
#                 classifier="SVC", n_components='all', dimReduction='PCA', 
#                 pre_speaker=False, nbr_elements=10000)
#    acc_ST_true[0] = generate_plots(dataset_name, one_lang=False, language="L2", 
#                 classifier="SVC", n_components='all', dimReduction='PCA', 
#                 pre_speaker=True, nbr_elements=10000)
#    acc_ST_false[0] = generate_plots(dataset_name, one_lang=False, language="L2", 
#                 classifier="SVC", n_components='all', dimReduction='PCA', 
#                 pre_speaker=False, nbr_elements=10000)
#    
#    
#    acc_mfcc_true[1] = generate_plots(dataset_name_MFCC, one_lang=False, language="L2", 
#                 classifier='kNN', n_components='all', dimReduction='PCA', 
#                 pre_speaker=True, nbr_elements=10000)
#    acc_mfcc_false[1] = generate_plots(dataset_name_MFCC, one_lang=False, language="L2", 
#                 classifier='kNN', n_components='all', dimReduction='PCA', 
#                 pre_speaker=False, nbr_elements=10000)
#    acc_ST_true[1] = generate_plots(dataset_name, one_lang=False, language="L2", 
#                 classifier='kNN', n_components='all', dimReduction='PCA', 
#                 pre_speaker=True, nbr_elements=10000)
#    acc_ST_false[1] = generate_plots(dataset_name, one_lang=False, language="L2", 
#                 classifier='kNN', n_components='all', dimReduction='PCA', 
#                 pre_speaker=False, nbr_elements=10000)
#    
#    
#    acc_mfcc_true[2] = generate_plots(dataset_name_MFCC, one_lang=False, language="L2", 
#                 classifier='tree', n_components='all', dimReduction='PCA', 
#                 pre_speaker=True, nbr_elements=10000)
#    acc_mfcc_false[2] = generate_plots(dataset_name_MFCC, one_lang=False, language="L2", 
#                 classifier='tree', n_components='all', dimReduction='PCA', 
#                 pre_speaker=False, nbr_elements=10000)
#    acc_ST_true[2] = generate_plots(dataset_name, one_lang=False, language="L2", 
#                 classifier='tree', n_components='all', dimReduction='PCA', 
#                 pre_speaker=True, nbr_elements=10000)
#    acc_ST_false[2] = generate_plots(dataset_name, one_lang=False, language="L2", 
#                 classifier='tree', n_components='all', dimReduction='PCA', 
#                 pre_speaker=False, nbr_elements=10000)
    
    
    
##    #Generate plots for FR train FR test
#    acc_mfcc_true = np.zeros(3)
#    acc_mfcc_false = np.zeros(3)
#    acc_ST_true = np.zeros(3)
#    acc_ST_false = np.zeros(3)
#    
#    acc_mfcc_true[0] = generate_plots(dataset_name_MFCC, one_lang=True, language="L2", 
#                 classifier="SVC", n_components='all', dimReduction='PCA', 
#                 pre_speaker=True, nbr_elements=10000)
#    acc_mfcc_false[0] = generate_plots(dataset_name_MFCC, one_lang=True, language="L2", 
#                 classifier="SVC", n_components='all', dimReduction='PCA', 
#                 pre_speaker=False, nbr_elements=10000)
#    acc_ST_true[0] = generate_plots(dataset_name, one_lang=True, language="L2", 
#                 classifier="SVC", n_components='all', dimReduction='PCA', 
#                 pre_speaker=True, nbr_elements=10000)
#    acc_ST_false[0] = generate_plots(dataset_name, one_lang=True, language="L2", 
#                 classifier="SVC", n_components='all', dimReduction='PCA', 
#                 pre_speaker=False, nbr_elements=10000)
#    
#    
#    acc_mfcc_true[1] = generate_plots(dataset_name_MFCC, one_lang=True, language="L2", 
#                 classifier='kNN', n_components='all', dimReduction='PCA', 
#                 pre_speaker=True, nbr_elements=10000)
#    acc_mfcc_false[1] = generate_plots(dataset_name_MFCC, one_lang=True, language="L2", 
#                 classifier='kNN', n_components='all', dimReduction='PCA', 
#                 pre_speaker=False, nbr_elements=10000)
#    acc_ST_true[1] = generate_plots(dataset_name, one_lang=True, language="L2", 
#                 classifier='kNN', n_components='all', dimReduction='PCA', 
#                 pre_speaker=True, nbr_elements=10000)
#    acc_ST_false[1] = generate_plots(dataset_name, one_lang=True, language="L2", 
#                 classifier='kNN', n_components='all', dimReduction='PCA', 
#                 pre_speaker=False, nbr_elements=10000)
#    
#    
#    acc_mfcc_true[2] = generate_plots(dataset_name_MFCC, one_lang=True, language="L2", 
#                 classifier='tree', n_components='all', dimReduction='PCA', 
#                 pre_speaker=True, nbr_elements=10000)
#    acc_mfcc_false[2] = generate_plots(dataset_name_MFCC, one_lang=True, language="L2", 
#                 classifier='tree', n_components='all', dimReduction='PCA', 
#                 pre_speaker=False, nbr_elements=10000)
#    acc_ST_true[2] = generate_plots(dataset_name, one_lang=True, language="L2", 
#                 classifier='tree', n_components='all', dimReduction='PCA', 
#                 pre_speaker=True, nbr_elements=10000)
#    acc_ST_false[2] = generate_plots(dataset_name, one_lang=True, language="L2", 
#                 classifier='tree', n_components='all', dimReduction='PCA', 
#                 pre_speaker=False, nbr_elements=10000)


#    #Generate plots for EN train EN test
    acc_mfcc_true = np.zeros(3)
    acc_mfcc_false = np.zeros(3)
    acc_ST_true = np.zeros(3)
    acc_ST_false = np.zeros(3)
    
#    acc_mfcc_true[0] = generate_plots(dataset_name_MFCC, one_lang=True, language="L1", 
#                 classifier="SVC", n_components='all', dimReduction='PCA', 
#                 pre_speaker=True, nbr_elements=10000)
#    acc_mfcc_false[0] = generate_plots(dataset_name_MFCC, one_lang=True, language="L1", 
#                 classifier="SVC", n_components='all', dimReduction='PCA', 
#                 pre_speaker=False, nbr_elements=10000)
    acc_ST_true[0] = generate_plots(dataset_name, one_lang=True, language="L1", 
                 classifier="SVC", n_components='all', dimReduction='PCA', 
                 pre_speaker=True, nbr_elements=10000)
    acc_ST_false[0] = generate_plots(dataset_name, one_lang=True, language="L1", 
                 classifier="SVC", n_components='all', dimReduction='PCA', 
                 pre_speaker=False, nbr_elements=10000)
    
    
    acc_mfcc_true[1] = generate_plots(dataset_name_MFCC, one_lang=True, language="L1", 
                 classifier='kNN', n_components='all', dimReduction='PCA', 
                 pre_speaker=True, nbr_elements=10000)
    acc_mfcc_false[1] = generate_plots(dataset_name_MFCC, one_lang=True, language="L1", 
                 classifier='kNN', n_components='all', dimReduction='PCA', 
                 pre_speaker=False, nbr_elements=10000)
    acc_ST_true[1] = generate_plots(dataset_name, one_lang=True, language="L1", 
                 classifier='kNN', n_components='all', dimReduction='PCA', 
                 pre_speaker=True, nbr_elements=10000)
    acc_ST_false[1] = generate_plots(dataset_name, one_lang=True, language="L1", 
                 classifier='kNN', n_components='all', dimReduction='PCA', 
                 pre_speaker=False, nbr_elements=10000)
    
    
    acc_mfcc_true[2] = generate_plots(dataset_name_MFCC, one_lang=True, language="L1", 
                 classifier='tree', n_components='all', dimReduction='PCA', 
                 pre_speaker=True, nbr_elements=10000)
    acc_mfcc_false[2] = generate_plots(dataset_name_MFCC, one_lang=True, language="L1", 
                 classifier='tree', n_components='all', dimReduction='PCA', 
                 pre_speaker=False, nbr_elements=10000)
    acc_ST_true[2] = generate_plots(dataset_name, one_lang=True, language="L1", 
                 classifier='tree', n_components='all', dimReduction='PCA', 
                 pre_speaker=True, nbr_elements=10000)
    acc_ST_false[2] = generate_plots(dataset_name, one_lang=True, language="L1", 
                 classifier='tree', n_components='all', dimReduction='PCA', 
                 pre_speaker=False, nbr_elements=10000)


    
    
#    # data to plot
#    n_groups = 3
#    #MFCC
#    acc = (0.811172, 0.670310, 0.572464)
#    pre = (0.81, 0.71, 0.58)
#    rec = (0.81, 0.67, 0.57)
#    f1  = (0.81, 0.66, 0.57)
#    #Speaker Trait
##    acc = [0.835591,0.720581, 0.9144]
##    pre = (0.84, 0.72, 0.91)
##    rec = (0.84, 0.72, 0.91)
##    f1 = (0.84, 0.72, 0.91)
#     
#    # create plot
#    fig, ax = plt.subplots()
#    index = np.arange(n_groups)
#    bar_width = 0.2
#    opacity = 0.8
#     
#    rects1 = plt.bar(index+bar_width/2-bar_width, acc, bar_width,
#                     alpha=opacity,
#                     color='r',
#                     label='Accuracy')
#     
#    rects2 = plt.bar(index+bar_width/2, pre, bar_width,
#                     alpha=opacity,
#                     color='g',
#                     label='Precision')
#    rects3 = plt.bar(index+bar_width/2+bar_width, rec, bar_width,
#                     alpha=opacity,
#                     color='b',
#                     label='Recall')
#    rects4 = plt.bar(index+bar_width/2+2*bar_width, f1, bar_width,
#                     alpha=opacity,
#                     color='orange',
#                     label='F1')
#    
#    
#
#    plt.xlabel('Models')
#    plt.ylabel('Metrics')
#    plt.title('Models Comparison for Speaker Test feature set')
#    plt.xticks(index + bar_width, ('SVM', 'kNN', 'Tree'))
#    plt.legend(loc=4)
#     
#    plt.tight_layout()
#    plt.savefig(dataset_name_MFCC+'_Models_comparison.png')
#    plt.show()
    
    
#        
#    
#    # Load dataset 
#    # data.data(name) loads the dataset based on training on French
#    # and testing with the other language
#    
#    df_train, df_test = data.data(dataset_name, language,nbr_elements=10000)
#    df_train_2, df_test_2 = data.data(dataset_name, language,nbr_elements=10000)
#
#    # data_ONE_LANG(name, language="L2", frac_train=0.02)
#    # loads or create the dataset based on training on argument language
#    # with the specific 
#    # and testing with the other language
##    df_train, df_test = data.data_ONE_LANG(dataset_name_MFCC, language, frac_train) 
#    
#    if 'frameIndex' in df_train.columns.values.tolist():
#        #Database MFCC
#        columns = ['Unnamed: 0', 'frameIndex', 'frameTime']
#        df_train.drop(columns, inplace=True, axis=1)
#        df_test.drop(columns, inplace=True, axis=1)
#    elif 'Unnamed: 0.1' in df_train.columns.values.tolist():
#        #Database Speaker trait
#        columns = ['Unnamed: 0', 'Unnamed: 0.1']
#        df_train.drop(columns, inplace=True, axis=1)
#        df_test.drop(columns, inplace=True, axis=1)
#        
#    
#    df_train = df_train.fillna(0)#all NaN replaced by 0
#    df_test = df_test.fillna(0)#all NaN replaced by 0
#    
#    
#    if pre_speaker:
#        df_train, df_test = standardization_speaker(df_train, df_test)
#    else:
#        df_train=df_train.sample(frac=1,random_state=1)
#        df_test=df_test.sample(frac=1,random_state=1)
#    
#    
#    dataset = data.restructure_data(df_train)
#    dataset_test = data.restructure_data(df_test)
#    
#    for i in range(10, 40, )
#    acc, cnf_matrix = trainAndEvaluate(dataset, dataset_test, n_components, dimReduction, classifier="SVC", preproc_speaker=pre_speaker)
#    print(acc)
##     plot_un_and_normalized_cnf(cnf_matrix, dataset['target_names'], dataset_name+'_'+language+'_'+'train'+'_'+'LEFT_FOR_TEST'+'_'+dimReduction+'_'+str(n_components))
#    plot_un_and_normalized_cnf(cnf_matrix, dataset['target_names'], dataset_name_MFCC+'_'+language+'_'+'ONLY'+'_'+str(frac_train)+'_'+dimReduction+'_'+str(n_components))
#    
#
#    x = dataset_test['data']
#    
##    x = pd.concat([dataset['data'],dataset_test['data']])
#    
#    
#    
#    
#    from sklearn.decomposition import PCA
#    pca = PCA(n_components=2)
#    principalComponents = pca.fit_transform(x)
#    principalDf = pd.DataFrame(data = principalComponents
#             , columns = ['principal component 1', 'principal component 2'])
#
#    finalDf = pd.concat([principalDf, dataset_test['target']], axis = 1)
#    
#
#    fig = plt.figure(figsize = (8,8))
#    ax = fig.add_subplot(1,1,1) 
#    ax.set_xlabel('Principal Component 1', fontsize = 15)
#    ax.set_ylabel('Principal Component 2', fontsize = 15)
#    ax.set_title('2 component PCA', fontsize = 20)
#    targets = ['Sp1','Sp2','Sp3']
#    colors = ['r', 'g', 'b']
#    for target, color in zip(targets,colors):
#        indicesToKeep = finalDf['target_names'] == target
#        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                   , finalDf.loc[indicesToKeep, 'principal component 2']
#                   , c = color
#                   , s = 50)
#    ax.legend(targets)
#    ax.grid()
#    
#    
#    df_train = df_mfcc.sample(frac=0.02, random_state=1)
#    df_test = df_mfcc.drop(df_train.index)
#    
#    df_test = df_test.sample(frac=0.02, random_state=1)
#    
#    dataset = data.restructure_data(df_train)
#    dataset_test = data.restructure_data(df_test)
#    
#    dataset_original = copy.deepcopy(dataset['data'])
#    dataset_test_original = copy.deepcopy(dataset_test['data'])

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

#    ## SVM
#    dataset['data'] = dataset_original
#    dataset_test['data'] = dataset_test_original
#
#    dataset['data'] = dataset['data'].values.astype('float')
#    dataset_test['data'] =  dataset_test['data'].values.astype('float')
#
#    print(trainAndEvaluate(dataset, dataset_test, classifier='SVC'))
#    # 0.8343206219918549
#    ## kNN
#    dataset['data'] = dataset_original
#    dataset_test['data'] = dataset_test_original
#
#    dataset['data'] = dataset['data'].values.astype('float')
#    dataset_test['data'] =  dataset_test['data'].values.astype('float')
#
#    print(trainAndEvaluate(dataset, dataset_test, classifier='kNN'))
#    # 0.7051092188078489
#
#    ## kNN
#    dataset['data'] = dataset_original
#    dataset_test['data'] = dataset_test_original
#
#    dataset['data'] = dataset['data'].values.astype('float')
#    dataset_test['data'] =  dataset_test['data'].values.astype('float')
#
#    print(trainAndEvaluate(dataset, dataset_test, classifier='tree'))
#    # 0.5194372454646428


    # Read Data
    
    
    # Pre Processing
    
    
    # Dimensionality Reduction



if __name__ == "__main__":
    main()

