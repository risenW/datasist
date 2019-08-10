import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Datastats:

    def __init__(self, data):
        self.data = data


    def _space(self):
        print('-' * 100)
        print('\n')

    def _check_data(self):
        print('-' * 100)
        print('\n')
        
    def describe(self, data=None, name='', date_cols=None, show_categories=False, plot_missing=False):
        '''
        Calculates statistics and information about a data set. Information like
        shapes, size, number of categorical/numeric or date features, number of missing values
        data types of objects e.t.c

        Parameters:
        data: Pandas DataFrame
            The data to describe
        '''
        data = self.data if data is None else data

        ## Get categorical features
        cat_features = self.get_cat_feats(self.data)
        
        #Get numerical features
        num_features = self.get_num_feats(self.data)


        print('Shape of {} data set: {}'.format(name, data.shape))
        self._space()
        print('Size of {} data set: {}'.format(name, data.size))
        self._space()
        print('Data Types')
        print("Note: All Non-numerical features are identified as objects")
        print(data.dtypes)
        self._space()
        print('Numerical Features in Data set')
        print(num_features)
        self._space()
        print('Statistical Description of Numerical Columns')
        print(data.describe())
        self._space()
        print('Categorical Features in Data set')
        print(cat_features)
        self._space()
        print('Unique class Count of Categorical features')
        print(self.get_unique_counts(data))
        self._space()
        if show_categories:     
            print('Classes in Categorical Columns')
            print(structdata.class_in_cat_feature(data, cat_features))
            self._space()

        print('Missing Values in Data')
        print(self.display_missing(data))

        #Plots the missing values
        if plot_missing:
            plot_missing(data)



    def plot_missing(self, data=None):
        '''
        Plots the data as a collection of its points to show missing values
        '''
        data = self.data if data is None else data
        sns.heatmap(data.isnull(), cbar=True)
        plt.show()



    def get_cat_feats(self, data=None):
        '''
        Returns the categorical features in a data set
        '''
        data = self.data if data is None else data

        cat_features = []
        for col in data.columns:
            if data[col].dtypes == 'object':
                cat_features.append(col)

        return cat_features


    def get_num_feats(self, data=None):
        '''
        Returns the numerical features in a data set
        '''
        data = self.data if data is None else data

        num_features = []
        for col in data.columns:
            if data[col].dtypes != 'object':
                num_features.append(col)
        
        return num_features




    def get_unique_counts(self, data=None):
        '''Gets the unique count of elements in a data set'''

        data = self.data if data is None else data
        features = self.get_cat_feats(data)
        temp_len = []

        for feature in features:
            temp_len.append(len(data[feature].unique()))
            
        dic = list(zip(features, temp_len))
        dic = pd.DataFrame(dic, columns=['Feature', 'Unique Count'])
        return dic


    def display_missing(self, data=None, plot=True):
        data = self.data if data is None else data
        df = data.isna().sum()
        df = df.reset_index()
        df.columns = ['features', 'missing_counts']

        missing_percent = round((df['missing_counts'] / data.shape[0]) * 100, 2)
        df['missing_percent'] = missing_percent

        if plot:
            self.plot_missing()
            return df
        else:
            return df



