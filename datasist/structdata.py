'''
This module contains all functions relating to the cleaning and exploration of structured data sets; mostly in pandas format

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .visualizations import class_count, plot_missing
from IPython.display import display
from collections import Counter

def describe(data=None, name='', date_cols=None, show_categories=False, plot_missing=False):
    '''
    Calculates statistics and information about a data set. Information displayed are
    shapes, size, number of categorical/numeric/date features, missing values,
    dtypes of objects etc.

    Parameters:
    --------------------
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

    Returns:
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

    print('Random five data points')
    display(data.sample(5))
    _space()

    print('Last five data points')
    display(data.tail())
    _space()

    print('Shape of {} data set: {}'.format(name, data.shape))
    _space()

    print('Size of {} data set: {}'.format(name, data.size))
    _space()

    print('Data Types')
    print("Note: All Non-numerical features are identified as objects in pandas")
    display(pd.DataFrame(data.dtypes, columns=['Data Type']))
    _space()

    date_cols = get_date_cols(data)
    if len(date_cols) is not 0:
        print("Column(s) {} should be in Datetime format. Use the [to_date] function in datasist.feature_engineering to convert to Pandas Datetime format".format(date_cols))
        _space()

    print('Numerical Features in Data set')
    print(num_features)
    _space()

    print('Categorical Features in Data set')
    display(cat_features)
    _space()

    print('Statistical Description of Columns')
    display(data.describe())
    _space()
    
    print('Description of Categorical Features')
    if cat_features != None:
        display(data.describe(include=[np.object, pd.Categorical]).T)
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

    Returns:
    -------
        List
            A list of all the categorical features in a dataset.
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    cat_features = num_features = data.select_dtypes(include=['object']).columns

    return list(cat_features)


def get_num_feats(data=None):
    '''
    Returns the numerical features in a data set

    Parameters:
    -----------
        data: DataFrame or named Series 

    Returns:
    -------
        List:
            A list of all the numerical features in a dataset.
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    num_features = data.select_dtypes(exclude=['object', 'datetime64']).columns

    return list(num_features)



def get_date_cols(data=None):
    '''
    Returns the Datetime columns in a data set.

    Parameters
    ----------
        data: DataFrame or named Series

            Data set to infer datetime columns from.

        convert: bool, Default True

            Converts the inferred date columns to pandas DateTime type
    Returns:
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
        
    df = list(zip(features, temp_len))
    df = pd.DataFrame(df, columns=['Feature', 'Unique Count'])
    df = df.style.bar(subset=['Unique Count'], align='mid')
    return df


def display_missing(data=None, plot=False):
    '''
    Display missing values as a pandas dataframe.

    Parameters
    ----------
        data: DataFrame or named Series

        plot: bool, Default False

            Plots missing values in dataset as a heatmap
    
    Returns
    -------
        Matplotlib Figure:

            Heatmap plot of missing values
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    df = data.isna().sum()
    df = df.reset_index()
    df.columns = ['features', 'missing_counts']

    missing_percent = round((df['missing_counts'] / data.shape[0]) * 100, 1)
    df['missing_percent'] = missing_percent

    if plot:
        plot_missing(data)
        return df
    else:
        return df





def cat_summarizer(data, x=None, y=None, hue=None, palette='Set1', verbose=True):
    '''
    Helper function that gives a quick summary of a given column of categorical data

    Parameters:
    ---------------------------
        dataframe: pandas dataframe

        x: str.
            horizontal axis to plot the labels of categorical data, y would be the count.

        y: str. 
            vertical axis to plot the labels of categorical data, x would be the count.

        hue: str. i
            if you want to compare it another variable (usually the target variable)

        palette: array, list.

            Colour of the plot

    Returns:
    ----------------------
        Quick Stats of the data and also the count plot

    '''
    if x == None:
        column_interested = y
    else:
        column_interested = x
    series = data[column_interested]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.countplot(x=x, y=y, hue=hue, data=data, palette=palette)
    plt.show()
    

def join_train_and_test(data_train=None, data_test=None):
    '''
    Joins two data sets and returns a dictionary containing their sizes and the concatenated data.
    Used mostly before feature engineering to combine train and test set together.

    Parameter:
    ----------
        data_train: DataFrame, named series.

            First data usually called train date to join.

        data_test: DataFrame, named series.

            Second data set to join, usually called test.
    
    Returns:
    -------
        Tuple: Merged data, size of train and size of test
    '''

    n_train = data_train.shape[0]
    n_test = data_test.shape[0]
    all_data = pd.concat([data_train, data_test],sort=False).reset_index(drop=True)
    
    return all_data, n_train, n_test


def detect_outliers(data, n, features):
    '''
        Detect Rows with outliers.

        Parameters
        ----------
            data: DataFrame or named Series

            n: the bench mark for the number of allowable outliers in the columns.
            
            features: Specific columns you want to check for outliers and it accepts only numerical values.

        Returns
        -------
            The rows where outliers are present.
        '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")

    if features is None:
        raise ValueError("columns: Expecting features i.e columns of the dataset but got 'None'")

    if n is None:
        n = 2

    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(data[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(data[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = data[(data[col] < Q1 - outlier_step) | (data[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


def check_train_test_set(train_data, test_data, index=None, col=None):
    '''
    Checks the distribution of train and test for uniqueness in order to determine
    the best feature engineering strategy.
    
    Parameters:
    -------------------
        train_data: DataFrame

            The first data set to join

        test_data: DataFrame
             
             The second dataset to join

        index: Str, Default None

            An index column present in both dataset to be used in plotting
        
        col: Str, Default None

            A feature present in both dataset used in plotting

    
    '''
    print('There are {} training rows and {} test rows.'.format(train_data.shape[0], test_data.shape[0]))
    print('There are {} training columns and {} test columns.'.format(train_data.shape[1], test_data.shape[1]))
    
    if index:
        if train_data[index].nunique() == train_data.shape[0]:
            print('Id field is unique.')
        else:
            print('Id field is not unique')

        if len(np.intersect1d(train_data[index].values, test_data[index].values))== 0:
            print('Train and test sets have distinct Ids.') 
        else:
            print('Train and test sets IDs are the same.')
            _space()

        plt.plot(train_data.groupby(col).count()[[index]], 'o-', label='train')
        plt.plot(test_data.groupby(col).count()[[index]], 'o-', label='test')
        plt.title('Train and test instances overlap.')
        plt.legend(loc=0)
        plt.ylabel('number of records')
        plt.show()
        
    
def _space():
    print('\n')

        

def _match_date(data):
    '''
        Return a list of columns that matches the DateTime expression
    '''
    mask = data.sample(20).astype(str).apply(lambda x : x.str.match(r'(\d{2,4}-\d{2}-\d{2,4})+').all())
    return set(data.loc[:, mask].columns)
