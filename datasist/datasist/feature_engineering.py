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
    
    temp_data = data.copy()
    missing_percent = (temp_data.isna().sum() / temp_data.shape[0]) * 100
    cols_2_drop = missing_percent[missing_percent.values > percent].index
    #Drop missing values
    temp_data.drop(cols_2_drop, axis=1, inplace=True)

    return temp_data


def fill_missing_cats(data=None, cat_features = None, method='mode'):
    '''
    Fill missing values using categorical features [method].

    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")

    if cat_features is None:
        cat_features = get_cat_feats(data)

    if method is 'mode':
        for feat in cat_features:
            data[feat].fillna(data[feat].mode())

    return data


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
        data[features[i]] = data[features[i]].fillna(mean)

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


    

    