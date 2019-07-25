#import neccessary modules and libraries for analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datastats
import structdata
import warnings

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)



def get_date_info(data=None, date_feat=None, drop_date=True, concatenate=False):
    '''
    TODO UPdate Doc
    Returns the date information from a given date column
    
    '''
    df = pd.DataFrame()
    df["date" + date_feat]=pd.to_datetime(data[date_feat])
    df[date_feat + "_dayofweek"] = df["date" + date_feat].dt.dayofweek
    df[date_feat + "_dayofyear"] =  df["date" + date_feat].dt.dayofyear
    df[date_feat + "_dayofmonth"] = df["date" + date_feat].dt.day
    df[date_feat + "_hour"] = df["date" + date_feat].dt.hour
    df[date_feat + "_minute"] = df["date" + date_feat].dt.minute
    df[date_feat + "_is_weekend"] = df["date" + date_feat].apply( lambda x : 1 if x  in [5,6] else 0 )
    df[date_feat + "_year"] = df["date" + date_feat].dt.year
    df[date_feat + "_quarter"] = df["date" + date_feat].dt.quarter
    df[date_feat + "_month"] = df["date" + date_feat].dt.month
    
    
    if concatenate:
        df = pd.concat((data, df), axis=1)
        if drop_date:
            df.drop(["date" + date_feat, date_feat], axis=1, inplace=True)
    else:
        if drop_date:
            df.drop(["date" + date_feat], axis=1, inplace=True)
      
    return df


def describe_date(data, date_feat):
    '''
    Calculate statistics of the date feature
    '''

    df = get_date_info(data, date_feat)
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
        num_features = datastats.get_num_feats(data)

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


