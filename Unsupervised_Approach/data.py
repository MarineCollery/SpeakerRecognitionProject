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
            
def read_mfcc():
    frames = []
    for root, dirs, files in os.walk('MFCC'):
        for file_n in files:
            filename = os.path.join(root, file_n)
            df_mfcc = pd.read_csv(filename, ';')
            df_mfcc['name'] = file_n            
            
            if 'Francois' in file_n:
                df_mfcc['target_names'] = "Sp1"
            if 'Marine' in file_n:
                df_mfcc['target_names'] = "Sp2"
            if 'Younes' in file_n:
                df_mfcc['target_names'] = "Sp3"
                
            if 'English' in file_n:
                df_mfcc['language'] = "L1"
            if 'Francais' in file_n:
                df_mfcc['language'] = "L2"
            if 'Allemand' in file_n:
                df_mfcc['language'] = "L3"
            if 'Espagnol' in file_n:
                df_mfcc['language'] = "L4"
            
            
            frames.append(df_mfcc)

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
    
    
    return dataset


def main():
    
    path_file = open("mypath.txt", "r")
    path = path_file.read()
    print(path)
    path_file.close()
    
    os.chdir(path)
    
    df = pd.read_csv("Speaker_trait/results.csv", ';')
    create_label_name_lan(df)
    
    df_mfcc = read_mfcc()
    
    
    # dataset = restructure_data(df)
    # dataset_mfcc = restructure_data(df_mfcc)


    return df, df_mfcc
    # Read Data
    
    
    # Pre Processing
    
    
    # Dimensionality Reduction



if __name__ == "__main__":
    main()