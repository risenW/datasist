'''
This module contains all functions relating to the cleaning and exploration of structured data sets; mostly in pandas format

'''


import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .visualizations import class_count, plot_missing
from IPython.display import display


def describe(data=None, name='', date_cols=None, show_categories=False, plot_missing=False):
    '''
    Calculates statistics and information about a data set. Information displayed are
    shapes, size, number of categorical/numeric/date features, missing values,
    dtypes of objects etc.

    Parameters:
    data: Pandas DataFrame
        The data to describe.
    name: str, optional
        The name of the data set passed to the function.
    date_cols: list/series/array
         Date column names in the data set.
    show_categories: bool, default False
        Displays the unique classes and counts in each of the categorical feature in the data set.
    plot_missing: bool, default True
        Plots missing values as a heatmap

    Returns
    -------
    None
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
        class_count(data, cat_features)
        _space()

    print('Missing Values in Data')
    display(display_missing(data))

    #Plots the missing values
    if plot_missing:
        plot_missing(data)



def get_cat_feats(data=None):
    '''
    Returns the categorical features in a data set

    Parameters:
    -----------
    data: DataFrame or named Series 

    Returns
    -------
    List
        A list of all the categorical features in a dataset.
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

    Parameters:
    -----------
    data: DataFrame or named Series 

    Returns
    -------
    List
        A list of all the numerical features in a dataset.
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    num_features = []
    for col in data.columns:
        if data[col].dtypes != 'object':
            num_features.append(col)
    
    return num_features



def get_date_cols(data=None, convert=True):
    '''
    Returns the Datetime columns in a data set.

    Parameters
    ----------
    data: DataFrame or named Series
        Data set to infer datetime columns from.
    convert: bool, Default True
        Converts the inferred date columns to pandas DateTime type
    Returns
    -------
    List
        Date column names in the data set
    '''

    
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    #Get existing date columns in pandas Datetime64 format
    date_cols = set(data.dtypes[data.dtypes == 'datetime64[ns, UTC]'].index)
    #infer Date columns 
    date_cols = date_cols.union(_match_date(data))

    return date_cols



def get_unique_counts(data=None):
    '''
    Gets the unique count of categorical features in a data set.

    Parameters
    -----------
    data: DataFrame or named Series 

    Returns
    -------
    DataFrame or Series
        Unique value counts of the features in a dataset.
    
    '''

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

    Parameters
    ----------
    data: DataFrame or named Series
    plot: bool, Default True
        Plots missing values in dataset as a heatmap
    
    Returns
    -------
    Image:
        Heatmap plot of missing values
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

        

def _match_date(data):
    '''
    Return a list of columns that matches the DateTime expression
    '''
    mask = data.head().astype(str).apply(lambda x : x.str.match(r'(\d{2,4}-\d{2}-\d{2,4})+').all())
    return set(data.loc[:, mask].columns)