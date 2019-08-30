'''
This module contains all functions relating to feature engineering
'''

import pandas as pd
import numpy as np
from .structdata import get_cat_feats, get_num_feats, get_date_cols


def drop_missing(data=None, percent=99):
    '''
    Drops missing columns with [percent] of missing data.

    Parameters:
    data: Pandas DataFrame or Series.
    percent: float, Default 99
        Percentage of missing values to be in a column before it is eligible for removal.

    Returns:
        Pandas DataFrame or Series.
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")
    
    missing_percent = (data.isna().sum() / data.shape[0]) * 100
    cols_2_drop = missing_percent[missing_percent.values > percent].index
    #Drop missing values
    data.drop(cols_2_drop, axis=1, inplace=True)



def drop_redundant(data):
    '''
    Removes features with the same value in all cell. 
    Drops feature If Nan is the second unique class as well.
    Parameters:
        data: DataFrame or named series
    
    Returns:
        DataFrame or named series
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")
    
    #get columns
    cols_2_drop = _nan_in_class(data)
    data.drop(cols_2_drop, axis=1, inplace=True)
    
    

def _nan_in_class(data):
    cols = []
    for col in data.columns:
        if len(data[col].unique()) == 1:
            cols.append(col)

        if len(data[col].unique()) == 2:
            if np.nan in list(data[col].unique()):
                cols.append(col)

    return cols



def fill_missing_cats(data=None, cat_features=None, missing_encoding=None):
    '''
    Fill missing values using the mode of the categorical features.
    Parameters:
    ----------
    data: DataFrame or name Series.
        Data set to perform operation on.
    cat_features: List, Series, Array.
        categorical features to perform operation on. If not provided, we automatically infer the categoricals from the dataset.
    missing_encoding: List, Series, Array.
            Values used in place of missing. Popular formats are [-1, -999, -99, '', ' ']
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")

    if cat_features is None:
        cat_features = get_cat_feats(data)

    temp_data = data.copy()
    #change all possible missing values to NaN
    if missing_encoding is None:
        missing_encoding = ['', ' ', -99, -999]

    temp_data.replace(missing_encoding, np.NaN, inplace=True)
    
    for col in cat_features:
        most_freq = temp_data[col].mode()[0]
        temp_data[col] = temp_data[col].replace(np.NaN, most_freq)
    
    return temp_data


def fill_missing_num(data=None, features=None, method='mean'):
    '''
    fill missing values in numerical columns with specified [method] value
    Parameters:
    ----------
    data: DataFrame or name Series.
        The data set to fill
    features: list.
        List of columns to fill
    method: str, Default 'mean'
        method to use in calculating fill value.
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")
    
    if features is None:
        #get numerical features with missing values
        num_feats = get_num_feats(data)
        temp_data = data[num_feats].isna().sum()
        features = list(temp_data[num_feats][temp_data[num_feats] > 0].index)
        print("Found {} with missing values.".format(features))

    for feat in features:
        if method is 'mean':
            mean = data[feat].mean()
            data[feat].fillna(mean, inplace=True)
        elif method is 'median':
            median = data[feat].median()
            data[feat].fillna(median, inplace=True)
        elif method is 'mode':
            mode = data[feat].mode()[0]
            data[feat].fillna(mode, inplace=True)
   
    return "Filled all missing values successfully"




def create_balanced_data(data=None, target=None, categories=None, class_sizes=None, replacement=False ):
    '''
    Creates a balanced data set from an imbalanced one. Used in a classification task.

    Parameter:
    data: DataFrame, name series.
        The imbalanced dataset.
    target: str
        Name of the target column.
    categories: list
        Unique categories in the target column. If not set, we use infer the unique categories in the column.
    class_sizes: list
        Size of each specified class. Must be in order with categoriess parameter.
    replacement: bool, Default True.
        samples with or without replacement.
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")
    
    if target is None:
        raise ValueError("target: Expecting a String got 'None'")

    if categories is None:
        categories = list(data[target].unique())
    
    if class_sizes is None:
        #set size for each class to same value
        temp_val = int(data.shape[0] / len(data[target].unique()))
        class_sizes = [temp_val for _ in list(data[target].unique())]

    
    temp_data = data.copy()
    data_category = []
    data_class_indx = []
    
    #get data corrresponding to each of the categories
    for cat in categories: 
        data_category.append(temp_data[temp_data[target] == cat])
    
    #sample and get the index corresponding to each category
    for class_size, cat in zip(class_sizes, data_category):
        data_class_indx.append(cat.sample(class_size, replace=True).index)
        
    #concat data together
    new_data = pd.concat([temp_data.loc[indx] for indx in data_class_indx], ignore_index=True).sample(sum(class_sizes)).reset_index(drop=True)
    
    if not replacement:
        for indx in data_class_indx:
            temp_data.drop(indx, inplace=True)
            
        
    return new_data



def to_date(data):
    '''
    Automatically convert all date time columns to pandas Datetime format
    '''

    date_cols = get_date_cols(data)
    for col in date_cols:
        data[col] = pd.to_datetime(data[col])
    
    return data


    

    