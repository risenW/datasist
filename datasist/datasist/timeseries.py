'''
This module contains all functions relating to timeseries data

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .structdata import get_cat_feats, get_num_feats, get_date_cols



def get_date_feats(data=None, date_features=None, date_cols_to_return=None, drop_date_feature=True):
    '''
    TODO UPdate Doc
    Returns the date information from a given date column
    
    '''
    df = data.copy()

    for date_feature in date_features:
        #Convert date feature to Pandas DateTime
        df[date_feature]=pd.to_datetime(df[date_feature])

        #specify columns to return
        dict_dates = {  "dow":  df[date_feature].dt.weekday_name,
                        "doy":   df[date_feature].dt.dayofyear,
                        "dom": df[date_feature].dt.day,
                        "hr": df[date_feature].dt.hour,
                        "minute":   df[date_feature].dt.minute,
                        "is_wkd":  df[date_feature].apply( lambda x : 1 if x  in [5,6] else 0 ),
                        "yr": df[date_feature].dt.year,
                        "qtr":  df[date_feature].dt.quarter,
                        "mth": df[date_feature].dt.month
                    } 
        date_fts = ['dow', 'doy', 'dom', 'hr', 'minute', 'is_wkd', 'yr', 'qtr', 'mth']

        if date_cols_to_return is None:
            #return all features
            for dt_ft in date_fts:
                df[date_feature + '_' + dt_ft] = dict_dates[dt_ft]
        else:
            #Return only sepcified date features
            for dt_ft in date_cols_to_return:
                df[date_feature + '_' + dt_ft] = dict_dates[dt_ft]
    
    if drop_date_feature:
        df.drop(date_features, axis=1, inplace=True)

    return df





def extract_time(data=None, time_cols=None, subset=None, drop=True):
    '''
    Returns time information in a pandas dataframe as a new set of columns 
    added to the original data frame.
    For extracting DateTime features, use datasist.timeseries.get_date_feats function
    
    Parameters:
    -----------
    data: DataFrame or named Series
        The data set to extract time information from.
    time_cols: List, Array
        Name of time columns/features in data set.
    subset: List, Array
        Time features to return default to [hours, minutes and seconds].
    drop: bool, Default True
        Drops the original time features from the data set.

    Return:
    -------
        DataFrame or Series.
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")
    
    if time_cols is None:
        raise ValueError("time_cols: Expecting a list, series/ numpy1D array, got 'None'")
    
    df = data.copy()
    
    if subset is None:
        subset = ['hours', 'minutes', 'seconds']
    
    for time_col in time_cols:  
        #Convert time columns to pandas time delta
        df[time_col] = pd.to_timedelta(df[time_col])
        
        for val in subset:
            df[time_col + "_" + val] = df[time_col].dt.components[val]
        
    if drop:
        #Drop original time columns
        df.drop(time_cols, axis=1, inplace=True)
        
    return df



def describe_date(data, date_feature):
    '''
    Calculate statistics of the date feature
    '''

    df = get_date_feats(data, date_feature)
    print(df.describe())




def num_to_time(data=None, num_features=None,time_col=None, subplots=True, marker='.', figsize=(15,10), y_label='Daily Totals',save_fig=False, alpha=0.5, linestyle='None'):
    '''
    Plots all numeric features against the time. Interpreted as a time series plot

    Parameters:
    #TODO UPDATE DOC
    
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if num_features is None:
        num_features = get_num_feats(data)
        #remove the time_Col from num_features
        num_features.remove(time_col)

    if time_col is None:
        raise ValueError("time_col: Expecting a Datetime Series, got 'None'")

    #Make time_col the index
    data[time_col] = pd.to_datetime(data[time_col])
    #Set as time_col as DataFrame index
    data = data.set_index(time_col)
    
    if subplots:
        axes = data[num_features].plot(marker=marker,subplots=True, figsize=figsize, alpha=0.5, linestyle=linestyle) 
        for feature, ax in zip(num_features, axes):
            ax.set_ylabel(y_label)
            ax.set_title("Timeseries Plot of '{}'".format(time_col))
            if save_fig:
                plt.savefig('fig_timeseries_plot_against_{}'.format(feature))
            plt.show()
    else:
        for feature in num_features:
            fig = plt.figure()
            ax = fig.gca()
            axes = data[feature].plot(marker=marker,subplots=False, figsize=figsize, alpha=0.5, linestyle=linestyle, ax=ax) 
            plt.ylabel(feature)
            ax.set_title("Timeseries Plot of '{}' vs. '{}' ".format(time_col, feature))
            if save_fig:
                plt.savefig('fig_timeseries_plot_against_{}'.format(feature))
            plt.show()           



def box_time_plot(data=None, features=None, x=None, subplots=True, figsize=(12,10)):
    '''
    Makes a box plot of features against a specified column
    
    '''

    if subplots:
        fig, axes = plt.subplots(len(features), 1, figsize=figsize, sharex=True)
        for feature, ax in zip(features, axes):
            sns.boxplot(data=data, x=x, y=feature, ax=ax)
            ax.set_ylabel('Count')
            ax.set_title("Boxplot of '{}' vs. {} ".format(feature, x))
            plt.tight_layout()
            # Remove the automatic x-axis label from all but the bottom subplot
            if ax != axes[-1]:
                ax.set_xlabel('')
    else:
        for feature in features:
            fig = plt.figure(figsize=figsize) # define plot area
            ax = fig.gca() # define axis 
            sns.boxplot(data=data, x=x, y=feature, ax=ax)
            ax.set_ylabel('Count')
            ax.set_title("Boxplot of '{}' vs. {} ".format(feature, x))
            plt.tight_layout()
            ax.set_xlabel('')
            plt.show()



def set_date_index(data, date_feature):
    #Make time_col the index
    data[date_feature] = pd.to_datetime(data[date_feature])
    #Set as time_col as DataFrame index
    return data.set_index(date_feature)

