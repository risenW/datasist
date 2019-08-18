'''
This module contains all functions relating to the cleaning and exploration of structured data sets; mostly in pandas format

'''


import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .visualizations import class_in_cat_feature, plot_missing
from IPython.display import display


def describe(data=None, name='', date_cols=None, show_categories=False, plot_missing=False):
    '''
    Calculates statistics and information about a data set. Information like
    shapes, size, number of categorical/numeric or date features, number of missing values
    data types of objects e.t.c

    Parameters:
    data: Pandas DataFrame
        The data to describe
    '''
    
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    ## Get categorical features
    cat_features = get_cat_feats(data)
    
    #Get numerical features
    num_features = get_num_feats(data)

    print('First five data points')
    display(data.head())
    _space()

    print('Last five data points')
    display(data.tail())
    _space()

    print('Shape of {} data set: {}'.format(name, data.shape))
    _space()

    print('Size of {} data set: {}'.format(name, data.size))
    _space()

    print('Data Types')
    print("Note: All Non-numerical features are identified as objects")
    display(pd.DataFrame(data.dtypes, columns=['Data Type']))
    _space()

    print('Numerical Features in Data set')
    print(num_features)
    _space()

    print('Statistical Description of Columns')
    display(data.describe())
    _space()

    print('Categorical Features in Data set')
    display(cat_features)
    _space()
    
    print('Unique class Count of Categorical features')
    display(get_unique_counts(data))
    _space()

    if show_categories:     
        print('Classes in Categorical Columns')
        print("-"*30)
        class_in_cat_feature(data, cat_features)
        _space()

    print('Missing Values in Data')
    display(display_missing(data))

    #Plots the missing values
    if plot_missing:
        plot_missing(data)



def get_cat_feats(data=None):
    '''
    Returns the categorical features in a data set
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    cat_features = []
    for col in data.columns:
        if data[col].dtypes == 'object':
            cat_features.append(col)

    return cat_features


def get_num_feats(data=None):
    '''
    Returns the numerical features in a data set
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    num_features = []
    for col in data.columns:
        if data[col].dtypes != 'object':
            num_features.append(col)
    
    return num_features




def get_unique_counts(data=None):
    '''Gets the unique count of elements in a data set'''

    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    features = get_cat_feats(data)
    temp_len = []

    for feature in features:
        temp_len.append(len(data[feature].unique()))
        
    dic = list(zip(features, temp_len))
    dic = pd.DataFrame(dic, columns=['Feature', 'Unique Count'])
    dic = dic.style.bar(subset=['Unique Count'], align='mid')
    return dic


def display_missing(data=None, plot=True):
    '''
    Display missing values as a pandas dataframe.
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    df = data.isna().sum()
    df = df.reset_index()
    df.columns = ['features', 'missing_counts']

    missing_percent = round((df['missing_counts'] / data.shape[0]) * 100, 2)
    df['missing_percent'] = missing_percent

    if plot:
        plot_missing(data)
        return df
    else:
        return df


def _space():
    print('\n')

        