'''
This module contains all functions relating to time series data

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from .structdata import get_cat_feats, get_num_feats, get_date_cols



def extract_dates(data=None, date_cols=None, subset=None, drop=True):
    '''
    Extracts date information in a dataframe and append to the original data as new columns.
    For extracting only time features, use datasist.timeseries.extract_time function
    
    Parameters:
    -----------
        data: DataFrame or named Series

            The data set to extract date information from.

        date_cols: List, Array

            Name of date columns/features in data set.

        subset: List, Array

            Date features to return. One of:
            ['dow' ==> day of the week
            'doy' ==> day of the year
            'dom' ==> day of the month
            'hr' ==> hour
            'min', ==> minute
            'is_wkd' ==> is weekend?
            'yr' ==> year
            'qtr' ==> quarter
            'mth' ==> month ]

        drop: bool, Default True

            Drops the original date columns from the data set.

    Return:
    -------
        DataFrame or Series.
    '''

    df = data.copy()

    for date_col in date_cols:
        #Convert date feature to Pandas DateTime
        df[date_col ]= pd.to_datetime(df[date_col])

        #specify columns to return
        dict_dates = {  "dow":  df[date_col].dt.weekday_name,
                        "doy":   df[date_col].dt.dayofyear,
                        "dom": df[date_col].dt.day,
                        "hr": df[date_col].dt.hour,
                        "min":   df[date_col].dt.minute,
                        "is_wkd":  df[date_col].apply( lambda x : 1 if x  in [5,6] else 0 ),
                        "yr": df[date_col].dt.year,
                        "qtr":  df[date_col].dt.quarter,
                        "mth": df[date_col].dt.month
                    } 

        if subset is None:
            #return all features
            subset = ['dow', 'doy', 'dom', 'hr', 'min', 'is_wkd', 'yr', 'qtr', 'mth']
            for date_ft in subset:
                df[date_col + '_' + date_ft] = dict_dates[date_ft]
        else:
            #Return only sepcified date features
            for date_ft in subset:
                df[date_col + '_' + date_ft] = dict_dates[date_ft]
    #Drops original time columns from the dataset
    if drop:
        df.drop(date_cols, axis=1, inplace=True)

    return df



def extract_time(data=None, time_cols=None, subset=None, drop=True):
    '''
    Returns time information in a pandas dataframe as a new set of columns 
    added to the original data frame.
    For extracting DateTime features, use datasist.timeseries.extract_dates function
    
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


def get_time_elapsed(data=None, date_cols=None, by='s', col_name=None):
    '''
    Calculates the time elapsed between two specified date columns 
    and returns the value in either seconds (s), minute (m) or hours (h).
    
    Parameter:
    ----------
        data: DataFrame or name series.

            The data where the Date features are located
        
        data_col: List

            list of Date columns on which to calculate time elpased

        by: str

            specifies how time elapsed is calculated. Can be one of [h,m,s] corresponding to
            hour, minute and seconds respectively.
        
        col_name: str

            Name to use for the created column.

                
    Returns:
    --------
        Pandas DataFrame with new column for elapsed time.
    '''

    if date_cols is None:
        raise ValueError("date_col: Expecting a list of Date columns, got 'None'")
    
    if len(date_cols) != 2:
        raise ValueError("date_col: lenght of date_cols should be 2, got '{}'".format(len(date_cols)))
    
    by_mapping = {'h': 'hrs', 'm': 'mins', 's': 'secs'}

    if data is None:

        date1 = pd.to_datetime(date_cols[0])
        date2 = pd.to_datetime(date_cols[1])

        if col_name is None:
            col_name = 'time_elapsed_' + by_mapping[by]
            time_elapsed = (date1 - date2) / np.timedelta64(1,by) 
            return pd.DataFrame(time_elapsed, columns=[col_name])
        else:
            time_elapsed = (date1 - date2) / np.timedelta64(1,by) 
            return pd.DataFrame(time_elapsed, columns=[col_name])
    else:
        #convert to Pandas DateTime format
        df = data.copy()

        date1 = pd.to_datetime(df[date_cols[0]])
        date2 = pd.to_datetime(df[date_cols[1]])

        if col_name is None:
            col_name = by_mapping[by] + '_btw_' + date_cols[0] + '_' + date_cols[1]
            df[col_name] = (df[date_cols[0]] - df[date_cols[1]]) / np.timedelta64(1,by) 
            return df

        else:
            df[col_name] = (df[date_cols[0]] - df[date_cols[1]]) / np.timedelta64(1,by) 
            return df


def get_period_of_day(date_col=None):
    '''
    Returns a list of the time of the day as regards to mornings, afternoons or evenings. Hour of the day that falls
    between [0,1,2,3,4,5,6,7,8,9,10,11,12] are mapped to mornings, [13,14,15,16]] are mapped to afternoons and [17,18,19,20,21,22,23] are mapped to eveinings. 
    
    Parameter:
    ------------
        date_cols: Series, 1-D DataFrame

            The datetime feature

    Returns:
    ----------
        Series of mapped values
    
    '''

    if date_col is None:
        raise ValueError("date_cols: Expect a date columns, got 'None'")

    
    if date_col.dtype != np.int:
        
        date_col_hr = pd.to_datetime(date_col).dt.hour
        return date_col_hr.map(_map_hours)
    
    else:
        return date_col.map(_map_hours)




def describe_date(data=None, date_col=None):
    '''
    Calculate statistics of the date feature

    Parameter:
    ---------
        data: DataFrame or name series.

            The data to describe.

        data_col: str

            Name of date column to describe
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if date_col is None:
        raise ValueError("date_col: Expecting a string, got 'None'")


    df = extract_dates(data, date_col)
    display(df.describe())




def timeplot(data=None, num_cols=None, time_col=None, subplots=True, marker='.', 
                    figsize=(15,10), y_label='Daily Totals',save_fig=False, alpha=0.5, linestyle='None'):
    '''
    Plot all numeric features against the time column. Interpreted as a time series plot.

    Parameters:
    -----------
        data: DataFrame, Series.

            The data used in plotting.

        num_cols: list, 1-D array.

            Numerical columns in the data set. If not provided, we automatically infer them from the data set.

        time_col: str.

            The time column to plot numerical features against. We set this column as the index before plotting.

        subplots: bool, Default True.

            Uses matplotlib subplots to make plots.

        marker: str

            matplotlib supported marker to use in line decoration.

        figsize: tuple of ints, Default (15,10)

            The figure size of the plot.

        y_label: str.

            Name of the Y-axis.

        save_fig: bool, Default True

            Saves the figure to the current working directory.
        
    Returns:
    --------
        matplotlib figure
    
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if num_cols is None:
        num_cols = get_num_feats(data)
        #remove the time_Col from num_cols
        num_cols.remove(time_col)

    if time_col is None:
        raise ValueError("time_col: Expecting a string name of time column, got 'None'")

    #Make time_col the index
    data[time_col] = pd.to_datetime(data[time_col])
    #Set as time_col as DataFrame index
    data = data.set_index(time_col)
    
    if subplots:
        axes = data[num_cols].plot(marker=marker,subplots=True, figsize=figsize, alpha=0.5, linestyle=linestyle) 
        for feature, ax in zip(num_cols, axes):
            ax.set_ylabel(y_label)
            ax.set_title("Timeseries Plot of '{}'".format(time_col))
            if save_fig:
                plt.savefig('fig_timeseries_plot_against_{}'.format(feature))
            plt.show()
    else:
        for feature in num_cols:
            fig = plt.figure()
            ax = fig.gca()
            axes = data[feature].plot(marker=marker,subplots=False, figsize=figsize, alpha=0.5, linestyle=linestyle, ax=ax) 
            plt.ylabel(feature)
            ax.set_title("Timeseries Plot of '{}' vs. '{}' ".format(time_col, feature))
            if save_fig:
                plt.savefig('fig_timeseries_plot_against_{}'.format(feature))
            plt.show()           



# def time_boxplot(data=None, features=None, x=None, subplots=True, figsize=(12,10)):
#     '''
#     Makes a box plot of features against a specified column
    
#     '''

#     if subplots:
#         fig, axes = plt.subplots(len(features), 1, figsize=figsize, sharex=True)
#         for feature, ax in zip(features, axes):
#             sns.boxplot(data=data, x=x, y=feature, ax=ax)
#             ax.set_ylabel('Count')
#             ax.set_title("Boxplot of '{}' vs. {} ".format(feature, x))
#             plt.tight_layout()
#             # Remove the automatic x-axis label from all but the bottom subplot
#             if ax != axes[-1]:
#                 ax.set_xlabel('')
#     else:
#         for feature in features:
#             fig = plt.figure(figsize=figsize) # define plot area
#             ax = fig.gca() # define axis 
#             sns.boxplot(data=data, x=x, y=feature, ax=ax)
#             ax.set_ylabel('Count')
#             ax.set_title("Boxplot of '{}' vs. {} ".format(feature, x))
#             plt.tight_layout()
#             ax.set_xlabel('')
#             plt.show()



def set_date_index(data, date_col):
    #Make time_col the index
    data[date_col] = pd.to_datetime(data[date_col])
    #Set as time_col as DataFrame index
    return data.set_index(date_col)


def _map_hours(x):   
    if x in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
        return 'morning'
    elif x in [13,14,15,16]:
        return 'afternoon'
    else:
        return 'evening'