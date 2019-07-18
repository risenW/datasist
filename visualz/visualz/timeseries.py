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