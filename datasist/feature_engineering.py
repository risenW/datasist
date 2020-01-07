'''
This module contains all functions relating to feature engineering
'''
import datetime as dt
import re
import platform
import pandas as pd
import numpy as np

if platform.system() == "Darwin":
    import matplotlib as plt
    plt.use('TkAgg')
else:
    import matplotlib.pyplot as plt

import seaborn as sns

from .structdata import get_cat_feats, get_num_feats, get_date_cols
from dateutil.parser import parse


def drop_missing(data=None, percent=99):
    '''
    Drops missing columns with [percent] of missing data.

    Parameters:
    -------------------------
        data: Pandas DataFrame or Series.

        percent: float, Default 99

            Percentage of missing values to be in a column before it is eligible for removal.

    Returns:
    ------------------
        Pandas DataFrame or Series.
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")
    
    missing_percent = (data.isna().sum() / data.shape[0]) * 100
    cols_2_drop = missing_percent[missing_percent.values >= percent].index
    print("Dropped {}".format(list(cols_2_drop)))
    #Drop missing values
    df = data.drop(cols_2_drop, axis=1)
    return df



def drop_redundant(data):
    '''
    Removes features with the same value in all cell. Drops feature If Nan is the second unique class as well.

    Parameters:
    -----------------------------
        data: DataFrame or named series.
    
    Returns:

        DataFrame or named series.
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")
    
    #get columns
    cols_2_drop = _nan_in_class(data)
    print("Dropped {}".format(cols_2_drop))
    df = data.drop(cols_2_drop, axis=1)
    return df
    


def fill_missing_cats(data=None, cat_features=None, missing_encoding=None, missing_col=False):
    '''
    Fill missing values using the mode of the categorical features.

    Parameters:
    ------------------------
        data: DataFrame or name Series.

            Data set to perform operation on.

        cat_features: List, Series, Array.

            categorical features to perform operation on. If not provided, we automatically infer the categoricals from the dataset.

        missing_encoding: List, Series, Array.

            Values used in place of missing. Popular formats are [-1, -999, -99, '', ' ']

        missin_col: bool, Default True

            Creates a new column to capture the missing values. 1 if missing and 0 otherwise. This can sometimes help a machine learning model.
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")

    if cat_features is None:
        cat_features = get_cat_feats(data)

    df = data.copy()
    #change all possible missing values to NaN
    if missing_encoding is None:
        missing_encoding = ['', ' ', -99, -999]

    df.replace(missing_encoding, np.NaN, inplace=True)
    
    for feat in cat_features:
        if missing_col:
            df[feat + '_missing_value'] = (df[feat].isna()).astype('int64')
        most_freq = df[feat].mode()[0]
        df[feat] = df[feat].replace(np.NaN, most_freq)
    
    return df


def fill_missing_num(data=None, num_features=None, method='mean', missing_col=False):
    '''
    fill missing values in numerical columns with specified [method] value

    Parameters:
        ------------------------------
        data: DataFrame or name Series.

            The data set to fill

        features: list.

            List of columns to fill

        method: str, Default 'mean'.

            method to use in calculating fill value.

        missing_col: bool, Default True

            Creates a new column to capture the missing values. 1 if missing and 0 otherwise. This can sometimes help a machine learning model.
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")
    
    if num_features is None:
        num_features = get_num_feats(data)
        #get numerical features with missing values
        temp_df = data[num_features].isna().sum()
        features = list(temp_df[num_features][temp_df[num_features] > 0].index)
        
    df = data.copy()
    for feat in features:
        if missing_col:
            df[feat + '_missing_value'] = (df[feat].isna()).astype('int64')
        if method is 'mean':
            mean = df[feat].mean()
            df[feat].fillna(mean, inplace=True)
        elif method is 'median':
            median = df[feat].median()
            df[feat].fillna(median, inplace=True)
        elif method is 'mode':
            mode = df[feat].mode()[0]
            df[feat].fillna(mode, inplace=True)
        else:
            raise ValueError("method: must specify a fill method, one of [mean, mode or median]'")

    return df


   


def merge_groupby(data=None, cat_features=None, statistics=None, col_to_merge=None):
    '''
    Performs a groupby on the specified categorical features and merges
    the result to the original dataframe.

    Parameter:
    -----------------------

        data: DataFrame

            Data set to perform operation on.

        cat_features: list, series, 1D-array

            categorical features to groupby.

        statistics: list, series, 1D-array, Default ['mean', 'count]

            aggregates to perform on grouped data.

        col_to_merge: str

            The column to merge on the dataset. Must be present in the data set.

    Returns:

        Dataframe.

    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")
    
    if statistics is None:     
        statistics = ['mean', 'count']
    
    if cat_features is None:
        cat_features = get_num_feats(data)

    if col_to_merge is None:
        raise ValueError("col_to_merge: Expecting a string [column to merge on], got 'None'")

    
    df = data.copy()
    
    for cat in cat_features:      
        temp = df.groupby([cat]).agg(statistics)[col_to_merge]
        #rename columns
        temp = temp.rename(columns={'mean': cat + '_' + col_to_merge + '_mean', 'count': cat + '_' + col_to_merge +  "_count"})
        #merge the data sets
        df = df.merge(temp, how='left', on=cat)
    
    
    return df


def get_qcut(data=None, col=None, q=None, duplicates='drop', return_type='float64'):
    '''
    Cuts a series into bins using the pandas qcut function
    and returns the resulting bins as a series for merging.

    Parameter:
    -------------

        data: DataFrame, named Series

            Data set to perform operation on.

        col: str

            column to cut/binnarize.

        q: integer or array of quantiles

            Number of quantiles. 10 for deciles, 4 for quartiles, etc. Alternately array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles.

        duplicates: Default 'drop',

            If bin edges are not unique drop non-uniques.

        return_type: dtype, Default (float64)

            Dtype of series to return. One of [float64, str, int64]
    
    Returns:
    --------

        Series, 1D-Array

    '''

    temp_df = pd.qcut(data[col], q=q, duplicates=duplicates).to_frame().astype('str')
    #retrieve only the qcut categories
    df = temp_df[col].str.split(',').apply(lambda x: x[0][1:]).astype(return_type)
    
    return df


def create_balanced_data(data=None, target=None, categories=None, class_sizes=None, replacement=False ):
    '''
    Creates a balanced data set from an imbalanced one. Used in a classification task.

    Parameter:
    ----------------------------
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

    
    df = data.copy()
    data_category = []
    data_class_indx = []
    
    #get data corrresponding to each of the categories
    for cat in categories: 
        data_category.append(df[df[target] == cat])
    
    #sample and get the index corresponding to each category
    for class_size, cat in zip(class_sizes, data_category):
        data_class_indx.append(cat.sample(class_size, replace=True).index)
        
    #concat data together
    new_data = pd.concat([df.loc[indx] for indx in data_class_indx], ignore_index=True).sample(sum(class_sizes)).reset_index(drop=True)
    
    if not replacement:
        for indx in data_class_indx:
            df.drop(indx, inplace=True)
            
        
    return new_data



def to_date(data):
    '''
    Automatically convert all date time columns to pandas Datetime format
    '''

    date_cols = get_date_cols(data)
    for col in date_cols:
        data[col] = pd.to_datetime(data[col])
    
    return data


def haversine_distance(lat1, long1, lat2, long2):
    '''
    Calculates the Haversine distance between two location with latitude and longitude.
    The haversine distance is the great-circle distance between two points on a sphere given their longitudes and latitudes.
    
    Parameter:
    ---------------------------
        lat1: scalar,float

            Start point latitude of the location.

        lat2: scalar,float 

            End point latitude of the location.

        long1: scalar,float

            Start point longitude of the location.

        long2: scalar,float 

            End point longitude of the location.

    Returns: 

        Series: The Harversine distance between (lat1, lat2), (long1, long2)
    
    '''

    lat1, long1, lat2, long2 = map(np.radians, (lat1, long1, lat2, long2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = long2 - long1
    distance = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    harvesine_distance = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(distance))
    harvesine_distance_df = pd.Series(harvesine_distance)
    return harvesine_distance_df


def manhattan_distance(lat1, long1, lat2, long2):
    '''
    Calculates the Manhattan distance between two points.
    It is the sum of horizontal and vertical distance between any two points given their latitudes and longitudes. 

    Parameter:
    -------------------
        lat1: scalar,float

            Start point latitude of the location.

        lat2: scalar,float 

            End point latitude of the location.

        long1: scalar,float

            Start point longitude of the location.

        long2: scalar,float 

            End point longitude of the location.

    Returns: Series

        The Manhattan distance between (lat1, lat2) and (long1, long2)
    
    '''
    a = np.abs(lat2 -lat1)
    b = np.abs(long1 - long2)
    manhattan_distance = a + b
    manhattan_distance_df = pd.Series(manhattan_distance)
    return manhattan_distance_df
    

def bearing(lat1, long1, lat2, long2):
    '''
    Calculates the Bearing  between two points.
    The bearing is the compass direction to travel from a starting point, and must be within the range 0 to 360. 

    Parameter:
    -------------------------
        lat1: scalar,float

            Start point latitude of the location.

        lat2: scalar,float 

            End point latitude of the location.

        long1: scalar,float

            Start point longitude of the location.

        long2: scalar,float 

            End point longitude of the location.

    Returns: Series

        The Bearing between (lat1, lat2) and (long1, long2)
    
    '''
    AVG_EARTH_RADIUS = 6371
    long_delta = np.radians(long2 - long1)
    lat1, long1, lat2, long2 = map(np.radians, (lat1, long1, lat2, long2))
    y = np.sin(long_delta) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(long_delta)
    bearing = np.degrees(np.arctan2(y, x))
    bearing_df = pd.Series(bearing)
    return bearing_df
    

def get_location_center(point1, point2):
    '''
    Calculates the center between two points.

    Parameter:
    ---------------------------
        point1: list, series, scalar

            End point latitude of the location.

        long1: list, series, scalar

            Start point longitude of the location.

        long2: list, series, scalar

            End point longitude of the location.

    Returns: Series
    
        The center between point1 and point2
    
    '''
    center = (point1 + point2) / 2
    center_df = pd.Series(center)
    return center_df

def log_transform(data, columns, plot=True, figsize=(12,6)):
    '''
    Nomralizes the dataset to be as close to the gaussian distribution.

    Parameter:
    -----------------------------------------
    data: DataFrame, Series.

        Data to Log transform.

    columns: List, Series

        Columns to be transformed to normality using log transformation
    
    plot: bool, default True

        Plots a before and after log transformation plot
    
    Returns:

        Log-transformed dataframe
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")

    if columns is None:
        raise ValueError("columns: Expecting at least a column in the list of columns but got 'None'")
    
    df = data.copy()
    for col in columns:
        df[col] = np.log1p(df[col])

    if plot:
        for col in columns: 
            _ = plt.figure(figsize = figsize)
            plt.subplot(1, 2, 1)
            sns.distplot(data[col], color="m", label="Skewness : %.2f" % (df[col].skew()))    
            plt.title('Distribution of ' + col + " before Log transformation")
            plt.legend(loc='best')
            
            plt.subplot(1, 2, 2)
            sns.distplot(df[col], color="m", label="Skewness : %.2f" % (df[col].skew()))    
            plt.title('Distribution of ' + col + " after Log transformation")
            plt.legend(loc='best')
            plt.tight_layout(2)
            plt.show()

    return df


def convert_dtype(df):
    '''
    Convert datatype of a feature to its original datatype.
    If the datatype of a feature is being represented as a string while the initial datatype is an integer or a float 
    or even a datetime dtype. The convert_dtype() function iterates over the feature(s) in a pandas dataframe and convert the features to their appropriate datatype
    
    Parameter:
    ---------------------------
    df: DataFrame, Series

        Dataset to convert data type
    
    Returns:
    -----------------
        DataFrame or Series.

    Example: 
    data = {'Name':['Tom', 'nick', 'jack'], 
            'Age':['20', '21', '19'],
            'Date of Birth': ['1999-11-17','20 Sept 1998','Wed Sep 19 14:55:02 2000']} 
     
    df = pd.DataFrame(data)

    df.info()
    >>> 
    <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 3 entries, 0 to 2
        Data columns (total 3 columns):
        Name             3 non-null object
        Age              3 non-null object
        Date of Birth    3 non-null object
        dtypes: object(3)
        memory usage: 76.0+ bytes
    
    conv = convert_dtype(df)
    conv.info()
    >>> 
    <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 3 entries, 0 to 2
        Data columns (total 3 columns):
        Name             3 non-null object
        Age              3 non-null int32
        Date of Birth    3 non-null datetime64[ns]
        dtypes: datetime64[ns](1), int32(1), object(1)
        memory usage: 88.0+ bytes


    '''
    if df.isnull().any().any() == True:
        raise ValueError("DataFrame contain missing values")
    else:
        i = 0
        changed_dtype = []
        #Function to handle datetime dtype
        def is_date(string, fuzzy=False):
            try:
                parse(string, fuzzy=fuzzy)
                return True
            except ValueError:
                return False
            
        while i <= (df.shape[1])-1:
            val = df.iloc[:,i]
            if str(val.dtypes) =='object':
                val = val.apply(lambda x: re.sub(r"^\s+|\s+$", "",x, flags=re.UNICODE)) #Remove spaces between strings
        
            try:
                if str(val.dtypes) =='object':
                    if val.min().isdigit() == True: #Check if the string is an integer dtype
                        int_v = val.astype(int)
                        changed_dtype.append(int_v)
                    elif val.min().replace('.', '', 1).isdigit() == True: #Check if the string is a float type
                        float_v = val.astype(float)
                        changed_dtype.append(float_v)
                    elif is_date(val.min(),fuzzy=False) == True: #Check if the string is a datetime dtype
                        dtime = pd.to_datetime(val)
                        changed_dtype.append(dtime)
                    else:
                        changed_dtype.append(val) #This indicate the dtype is a string
                else:
                    changed_dtype.append(val) #This could count for symbols in a feature
            
            except ValueError:
                raise ValueError("DataFrame columns contain one or more DataType")
            except:
                raise Exception()

            i = i+1

        data_f = pd.concat(changed_dtype,1)

        return data_f
            


def bin_age(data, feature, bins, labels, fill_missing = None, drop_original = False):

    '''
    Categorize age data into separate bins

    Parameter:
    -----------------------------------------
    data: DataFrame, Series.

        Data for which feature to be binned exist.

    feature: List, Series

        Columns to be binned
    
    Bins: List, numpy.ndarray

        Specifies the different categories. Bins must be one greater labels.
    
    labels: List, Series

        Name identified to the various categories

    fill_missing(default = None): int

        mean : feature average.
        mode : most occuring data in the feature.
        median : middle point in the feature.

    drop_original: bool

        Drops original feature after beaning.

    Returns:
        Returns a binned dataframe.
    '''

    
    df = data.copy()
    for col in feature:
        if fill_missing == None:
        
            if df[col].isnull().any():
                raise ValueError("data: Mising Value found in table")
            
            else:
                df[col + '_binned'] = pd.cut(x=df[col], bins= bins, labels=labels)
            
    
        elif fill_missing == 'mean':
            df[col].fillna(int(df[col].mean()), inplace  = True)
            df[col + '_binned'] = pd.cut(x=df[col], bins=bins, labels=labels)

        elif fill_missing == 'mode':
            df[col].fillna(int(df[col].mode()), inplace  = True)
            df[col + '_binned'] = pd.cut(x=df[col], bins=bins, labels=labels)
    
        elif fill_missing == 'median':
            df[col].fillna(int(df[col].median()), inplace  = True)
            df[col + '_binned'] = pd.cut(x=df[col], bins=bins, labels=labels)
            
        
        if drop_original == True:
           
            df.drop(columns = col, inplace = True)

    return df
    


def _nan_in_class(data):
    cols = []
    for col in data.columns:
        if len(data[col].unique()) == 1:
            cols.append(col)

        if len(data[col].unique()) == 2:
            if np.nan in list(data[col].unique()):
                cols.append(col)

    return cols
