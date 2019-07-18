import numpy as numpy
import pandas as pd
import math 
import structdata



def describe(data, name='', show_categories=False, plot_missing=False):
    '''
    Calculates statistics and information about a data set. Information like
    shapes, size, number of categorical/numeric or date features, number of missing values
    data types of objects e.t.c

    Parameters:
    data: Pandas DataFrame
        The data to describe
    '''

    ## Get categorical features
    cat_features = get_cat_feats(data)
    
    #Get numerical features
    num_features = get_num_feats(data)

    #Get time features

    print('Shape of {} data set: {}'.format(name, data.shape))
    space()
    print('Size of {} data set: {}'.format(name, data.size))
    space()
    print('Data Types')
    print("Note: All Non-numerical features are identified as objects")
    print(data.dtypes)
    space()
    print('Numerical Features in Data set')
    print(num_features)
    space()
    print('Statistical Description of Numerical Columns')
    print(data.describe())
    space()
    print('Categorical Features in Data set')
    print(cat_features)
    space()
    print('Unique class Count of Categorical features')
    print(get_unique_counts(data))
    space()
    if show_categories:     
        print('Classes in Categorical Columns')
        print(structdata.class_in_cat_feature(data, cat_features))
        space()       

    print('Missing Values in Data')
    print(data.isna().sum())

    #Plots the missing values
    if plot_missing:
        plot_missing(data)





def plot_missing(data):
    '''
    Plots the data as a collection of its points to show missing values
    '''


def get_cat_feats(data):
    '''
    Returns the categorical features in a data set
    '''
    cat_features = []
    for col in data.columns:
        if data[col].dtypes == 'object':
            cat_features.append(col)

    return cat_features


def get_num_feats(data):
    '''
    Returns the numerical features in a data set
    '''
    num_features = []
    for col in data.columns:
        if data[col].dtypes != 'object':
            num_features.append(col)
    
    return num_features



def get_date_feats(data):
    '''
    Returns the date features in a data set
    '''

def get_unique_counts(data):
    '''Gets the unique count of elements in a data set'''

    features = get_cat_feats(data)
    temp_len = []
    for feature in features:
        temp_len.append(len(data[feature].unique()))
        
    dic = list(zip(features, temp_len))
    dic = pd.DataFrame(dic, columns=['Feature', 'Unique Count'])
    return dic


def space():
    print('-' * 100)
    print('\n')