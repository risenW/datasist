'''
This module contains all functions relating to feature engineering
'''

import pandas as pd
import numpy as np
from .structdata import get_cat_feats, get_num_feats, get_date_cols
from dateutil.parser import parse
import datetime as dt


def drop_missing(data=None, percent=99):
    '''
    Drops missing columns with [percent] of missing data.

    Parameters:
    -------------------------
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
    print("Dropped {}".format(list(cols_2_drop)))
    #Drop missing values
    data.drop(cols_2_drop, axis=1, inplace=True)



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
    ------------------------
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
        ------------------------------
        data: DataFrame or name Series.

            The data set to fill

        features: list.

            List of columns to fill

        method: str, Default 'mean'.

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

            Number of quantiles. 10 for deciles, 4 for quartiles, etc. Alternately
            array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles.

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

def convert_dtype(df):
    '''
    Convert datatype of a feature to its original datatype.
    E.g If the datatype of a feature is being represented as a string while the initial datatype is an integer or a float 
    or even a datetime dtype. The convert_dtype() function iterates over the feature(s) in a pandas dataframe and convert the features to their appropriate datatype
    For the function to work:
    1. There must be no missing values in the features
    2. The feature must contain only one default datatype in the feature. Must not contain 
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
            


    