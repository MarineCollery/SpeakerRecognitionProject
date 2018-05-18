#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:50:26 2018

@author: Marine
"""

import pandas as pd
import os

import numpy as np



# Creating target names
def create_label_name_lan(df):
    df['target_names'] = "name"
    df['language'] = "lan"
    for i in range(0,df['name'].size):
        df.loc[:,('name',i)]
        if 'Francois' in df.loc[i,('name')]:
            df.loc[i,'target_names'] = "Sp1"
        if 'Marine' in df.loc[i,('name')]:
            df.loc[i,'target_names'] = "Sp2"
        if 'Younes' in df.loc[i,('name')]:
            df.loc[i,'target_names'] = "Sp3"
            
        if 'English' in df.loc[i,('name')]:
            df.loc[i,'language'] = "L1"
        if 'Francais' in df.loc[i,('name')]:
            df.loc[i,'language'] = "L2"
        if 'Allemand' in df.loc[i,('name')]:
            df.loc[i,'language'] = "L3"
        if 'Espagnol' in df.loc[i,('name')]:
            df.loc[i,'language'] = "L4"
            


def read_features(name_folder):
    frames = []
    for root, dirs, files in os.walk(name_folder):
        for file_n in files:
            filename = os.path.join(root, file_n)
            df = pd.read_csv(filename, ';')
            df['name'] = file_n            
            
            if 'Francois' in file_n:
                df['target_names'] = "Sp1"
            if 'Marine' in file_n:
                df['target_names'] = "Sp2"
            if 'Younes' in file_n:
                df['target_names'] = "Sp3"
                
            if 'English' in file_n:
                df['language'] = "L1"
            if 'Francais' in file_n:
                df['language'] = "L2"
            if 'Allemand' in file_n:
                df['language'] = "L3"
            if 'Espagnol' in file_n:
                df['language'] = "L4"
             
            frames.append(df)

    final_df = pd.concat(frames)

    
    return final_df



def restructure_data(df):
    features = df.loc[:, df.columns != 'name']
    features = features.loc[:, features.columns != 'target_names']
    features = features.loc[:, features.columns != 'language']
    
    target = df['target_names']
    
    dataset = {}
    
    target_names, counts = np.unique(df['target_names'], return_counts=True)
    print("The data is classified like: {}".format(dict(zip(target_names, counts))))
    
    dataset['data'] = features
    dataset['target'] = target
    dataset['target_names'] = target_names
    dataset['language'] = df['language']
    
    
    return dataset


def split_data_balanced_ONLY_ONE_LANG(df, name, language="L2", frac_train=0.2):
    new_df = df[df.language==language]
    
    df_ref_SP1 = new_df[(new_df.target_names=="Sp1")]
    df_ref_SP2 = new_df[(new_df.target_names=="Sp2")]
    df_ref_SP3 = new_df[(new_df.target_names=="Sp3")]    
    

    nb_sample_max = min(df_ref_SP1.shape[0],df_ref_SP2.shape[0],df_ref_SP3.shape[0])
    nb_row_by_speaker = int(nb_sample_max*frac_train)

    print(nb_row_by_speaker)

    df_train = pd.concat([df_ref_SP1.sample(nb_row_by_speaker, random_state=12),
                        df_ref_SP2.sample(nb_row_by_speaker, random_state=12),
                        df_ref_SP3.sample(nb_row_by_speaker, random_state=12)])# number of samples we keep

    df_test = new_df.drop(df_train.index)
    #TO DO dOES NOT WORK !!!
    print(df_train.shape[0])
    print(df_test.shape[0])
    
    df_train.sample(frac=1)# randomize 
    df_test.sample(frac=1)# randomize 

    df_train.to_csv("TRAINING_"+name+"_speaker_balanced_"+language+"_training_"+language+"_testing_"+str(frac_train)+"_for_training.csv",index=True)
    df_test.to_csv("TESTING_"+name+"_speaker_balanced_"+language+"_training_"+language+"_testing_"+str(frac_train)+"_for_training.csv",index=True)
    
    return df_train, df


def split_data_balanced_LANGUAGE(df, name, language="L2"):
    print(df.size)
    frac_train=1
    df_ref_SP1 = df[(df.target_names=="Sp1") & (df.language==language)]
    df_ref_SP2 = df[(df.target_names=="Sp2") & (df.language==language)]
    df_ref_SP3 = df[(df.target_names=="Sp3") & (df.language==language)]
    df_ref_left = df[df.language!=language]
    
    df_ref_SP1_train = df_ref_SP1.sample(frac=frac_train, random_state=1)
    
    df_ref_SP2_train = df_ref_SP2.sample(frac=frac_train, random_state=1)
    
    df_ref_SP3_train = df_ref_SP3.sample(frac=frac_train, random_state=1)
    
    
    df_train = pd.concat([df_ref_SP1_train, df_ref_SP2_train, df_ref_SP3_train])
    df_test = df_ref_left
    
    df_train.sample(frac=1)# randomize
    df_test.sample(frac=1)# randomize

    print(df_train.size)
    print(df_test.size)
    
    df_train.to_csv("TRAINING_"+name+"_speaker_balanced_"+language+"_training_OTHER_testing.csv",index=True)
    df_test.to_csv("TESTING_"+name+"_speaker_balanced_"+language+"_training_OTHER_testing.csv",index=True)
    
    return df_train, df_test



def data(name, language="L2"):
    try:
        df_train = pd.read_csv("TRAINING_"+name+"_speaker_balanced_"+language+"_training_OTHER_testing.csv", ',')
        df_test = pd.read_csv("TESTING_"+name+"_speaker_balanced_"+language+"_training_OTHER_testing.csv", ',')
        print("database "+name+" already splitted and created")
        return df_train,df_test
    except:
        print("database "+name+" needs to be created")
        df = read_features(name)
        df_train, df_test = split_data_balanced_LANGUAGE(df, name, language)
        return df_train,df_test
    

def data_ONE_LANG(name, language="L2", frac_train=0.2):
    try:
        df_train = pd.read_csv("TRAINING_"+name+"_speaker_balanced_"+language+"_training_"+language+"_testing_"+str(frac_train)+"_for_training.csv", ',')
        df_test = pd.read_csv("TESTING_"+name+"_speaker_balanced_"+language+"_training_"+language+"_testing_"+str(frac_train)+"_for_training.csv", ',')
        print("database "+name+" already splitted and created")
        return df_train,df_test
    except:
        print("database "+name+" needs to be created")
        df = read_features(name)
        df_train, df_test = split_data_balanced_ONLY_ONE_LANG(df, name, language, frac_train)
        return df_train,df_test
    
    
def train_test_split_mfcc(df, frac_train, random=1):
    df_train = df.sample(frac=frac_train, random_state=random)
    df_test = df.drop(df_train.index)
    return df_train, df_test
    


def main():
    
    path_file = open("mypath.txt", "r")
    path = path_file.read()
    print(path)
    path_file.close()
    
    os.chdir(path)

    
    df_train,df_test = data('Speaker_trait', language="L2")
    df_train,df_test = data('MFCC', language="L2")
#    df_train_mfcc, df_test_mfcc = data('MFCC')
    
    
#    df_train2,df_test2 = data_ONE_LANG('Speaker_trait', language="L2", frac_train=0.2)

    
    # df_mfcc = read_features('MFCC')
    
#    df=restructure_data(df_train)
    
    
    
    
#    dataset = restructure_data(df)
#    dataset_mfcc = restructure_data(df_mfcc)
    
    # return df_train,df_test
    # Read Data
    
    
    # Pre Processing
    
    
    # Dimensionality Reduction



if __name__ == "__main__":
    main()