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


#TODO Update function to take different types of fill value
def fill_missing_num(data=None, features=None):
    '''
    fills all missing values in numerical columns 
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")
    
    if features is None:
         raise ValueError("features: Expected a list of columns")

    for i in range(len(features)):
        mean = data[features[i]].mean()
        data[features[i]] = data[features[i]].fillna(mean, inplace=True)

    return data



def create_balanced_data(data, target_name, target_cats=None, n_classes=None, replacement=False ):
    '''
    Creates a balanced data set from an imbalanced data
    Parameter:
    
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")
    
    if target_name is None:
        raise ValueError("target: Expecting a Series/ numpy1D array, got 'None'")
    
    temp_data = data.copy()
    new_data_size = sum(n_classes)
    classes = []
    class_index = []
    
    #get classes from data
    for t_cat in target_cats: 
        classes.append(temp_data[temp_data[target_name] == t_cat])
    
    for n_class, clas in zip(n_classes, classes):
        class_index.append(clas.sample(n_class, replace=True).index)
        
    #concat data together
    new_data = pd.concat([temp_data.loc[indx] for indx in class_index], ignore_index=True).sample(new_data_size).reset_index(drop=True)
    new_data_target = new_data[target_name]
    #drop new data target
    new_data.drop(target_name, axis=1, inplace=True)
    
    if not replacement:
        for indx in class_index:
            temp_data.drop(indx, inplace=True)
            
    #drop target from data
    original_target = temp_data[target_name]
    temp_data.drop(target_name, axis=1, inplace=True)
    
    print("shape of data {}".format(temp_data.shape))
    print("shape of data target {}".format(original_target.shape))
    print("shape of created data {}".format(new_data.shape))
    print("shape of created data target {}".format(new_data_target.shape))
    
    return temp_data, new_data, original_target, new_data_target



def to_date(data):
    '''
    Automatically convert all date time columns to pandas Datetime format
    '''

    date_cols = get_date_cols(data)
    for col in date_cols:
        data[col] = pd.to_datetime(data[col])
    
    return data


    

    