#import neccessary modules and libraries for analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics



def bar_cat_features(data=None, cat_features=None, fig_size=(5,5), save_fig=False):
    '''
    Makes a bar plot of all categorical features to show their counts.
    
    Parameters
    ------------

    data : Pandas dataframe.
    cat_features: Scalar, array, or list. 
               The categorical featureures in the dataset, if not provided, 
               we try to infer the categorical columns from the dataframe.
    fig_size: tuple
              The size of the figure object
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if cat_features is None:
        cat_features = []
        for col in data.columns:
            if data[col].dtypes == 'object':
                cat_features.append(col)
                

    for feature in cat_features:
        fig = plt.figure(figsize=fig_size)
        ax = fig.gca()

        #get the value count of the column
        v_count = data[feature].value_counts()
        v_count.plot.bar(ax = ax)
        plt.xticks(rotation=90)
        ax.set_title("Bar plot for " + feature)

        if save_fig:
            plt.savefig('Barplot_{}'.format(feature))





def box_num_2_cat_target(data=None, num_features=None, target=None, fig_size=(5,5), large_data=False, save_fig=False):
    '''
    Makes a box plot of all numerical featureures against a specified categorical target column.
 
    Parameters
    ------------

    data : Pandas dataframe.
    num_features: Scalar, array, or list. 
               The numerical featureures in the dataset, if not provided, 
               we try to infer the numerical columns from the dataframe.
    target: array, pandas series, list.
            A categorical target column. Maximun number of categories is 7 and minimum is 1
    fig_size: tuple
              The size of the figure object
    large_data: boolean
            If True, then sns boxenplot is used instead of normal boxplot. Boxenplot is 
            better for large dataset
    save_fig: boolean
            If True, saves the current plot to the current working directory
    '''

    if target is None:
        raise ValueError('Target value cannot be None')

    if len(data[target].unique()) > 7:
        raise AttributeError("Target categories must be less than seven")


    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if num_features is None:
        num_features = []
        for col in data.columns:
            if data[col].dtypes != 'object':
                num_features.append(col)
    
    if large_data:
        #use advanced sns boxenplot
         for feature in num_features:
            fig = plt.figure(figsize=fig_size)
            ax = fig.gca()

            sns.set_style("whitegrid")
            sns.boxenplot(target, feature, data=data, ax=ax)
            plt.xlabel(feature) # Set text for the x axis
            plt.ylabel(target)# Set text for y axis
            plt.xticks(rotation=90)
            plt.title('Box plot of {} against {}'.format(feature, target))
            if save_fig:
                plt.savefig('fig_{}_vs_{}'.format(feature,target))
            plt.show()
    else:
         for feature in num_features:
            fig = plt.figure(figsize=fig_size)
            ax = fig.gca()

            sns.set_style("whitegrid")
            sns.boxplot(target, feature, data=data, ax=ax)
            plt.xlabel(feature) # Set text for the x axis
            plt.ylabel(target)# Set text for y axis
            plt.xticks(rotation=90)
            plt.title("Box plot of '{}' vs. '{}'".format(feature, target))
            if save_fig:
                plt.savefig('fig_{}_vs_{}'.format(feature,target))
            plt.show()




def violin_num_2_cat_target(data=None, num_features=None, target=None, fig_size=(5,5), save_fig=False):
    '''
    Makes a violin plot of all numerical featureures against a specified categorical target column.
 
    Parameters
    ------------

    data : Pandas dataframe.
    num_features: Scalar, array, or list. 
               The numerical featureures in the dataset, if not provided, 
               we try to infer the numerical columns from the dataframe.
    target: array, pandas series, list.
            A categorical target column. Maximun number of categories is 7 and minimum is 1
    fig_size: tuple
              The size of the figure object
    save_fig: boolean
            If True, saves the current plot to the current working directory
    '''

    if target is None:
        raise ValueError('Target value cannot be None')

    if len(data[target].unique()) > 7:
        raise AttributeError("Target categories must be less than seven")


    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if num_features is None:
        num_features = []
        for col in data.columns:
            if data[col].dtypes != 'object':
                num_features.append(col)

    for feature in num_features:
        fig = plt.figure(figsize=fig_size)
        ax = fig.gca()

        sns.set_style("whitegrid")
        sns.violinplot(target, feature, data=data, ax=ax)
        plt.xticks(rotation=90)
        plt.xlabel(feature) # Set text for the x axis
        plt.ylabel(target)# Set text for y axis
        plt.title("Violin plot of '{}' vs. '{}'".format(feature, target))
        if save_fig:
            #TODO Add function to save to a specified directory
            plt.savefig('fig_{}_vs_{}'.format(feature,target))
        plt.show()





def hist_num_features(data=None, num_features=None, bins=5, show_dist_type=False, fig_size=(5,5), save_fig=False):
    '''
    Makes an histogram plot of all numerical featureures. Helps to show the distribution of the featureures.
    
 
    Parameters
    ------------

    data : Pandas dataframe.
    num_features: Scalar, array, or list. 
               The numerical featureures in the dataset, if not provided, 
               we try to infer the numerical columns from the dataframe.
    bins: int
            The number of bins to use.
    show_dist_type: boolean
            If True, Calculates the skewness of the data and display one of (Left skewed, right skewed or normal) 
    fig_size: tuple
              The size of the figure object
    save_fig: boolean
            If True, saves the current plot to the current working directory
    '''


    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if num_features is None:
        num_features = []
        for col in data.columns:
            if data[col].dtypes != 'object':
                num_features.append(col)

    for feature in num_features:
        fig = plt.figure(figsize=fig_size)
        ax = fig.gca()

        sns.distplot(data[feature], ax=ax, bins=bins)
        ax.set_xlabel(feature) # Set text for the x axis
        ax.set_ylabel('Count')# Set text for y axis

        if show_dist_type:
            ##TODO Add Code to calculate skewness
            pass
        else:
            ax.set_title('Histogram of ' + feature)

        if save_fig:
            #TODO Add function to save to a specified directory
            plt.savefig('fig_hist_{}'.format(feature))

        plt.show()






def bar_cat_2_cat_target(data=None, cat_features=None, target=None, fig_size=(12,6), save_fig=False):
    '''
    Makes a side by side bar plot of all categorical featureures against the target classes.
    
 
    Parameters
    ------------

    data : Pandas dataframe.
            The data we are working with.
    cat_features: Scalar, array, or list. 
               The categorical features in the dataset, if not provided, 
               we try to infer the categorical columns from the dataframe.
    fig_size: tuple
              The size of the figure object
    save_fig: boolean
            If True, saves the current plot to the current working directory
    '''


    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if cat_features is None:
        cat_features = []
        for col in data.columns:
            if data[col].dtypes == 'object':
                cat_features.append(col)

    #remove target from cat_features
    cat_features.remove(target)

    #Create a dummy column to hold count of values
    data['dummy_count'] = np.ones(shape = data.shape[0])
    #Loop over each categorical featureure and plot the acceptance rate for each category.
    for feature in cat_features:
        counts = data[['dummy_count', target, feature]].groupby([target, feature], as_index = False).count() #get the counts for each category
        #get the categories
        cats = list(data[target].unique())

        if len(cats) > 6:
            raise ValueError("Target column: '{}' must contain less than six unique classes".format(target))
        
        #create new figure
        _ = plt.figure(figsize = fig_size)

        for i, cat in enumerate(cats): 
            plt.subplot(1, len(cats), i+1)
            #Get the counts each category in target     
            temp = counts[counts[target] == cat][[feature, 'dummy_count']] 
            sns.barplot(x=feature, y='dummy_count', data=temp)
            plt.xticks(rotation=90)
            plt.title('Counts for {} \n class {}'.format(feature, cat))
            plt.ylabel('count')

            if save_fig:
                plt.savefig('fig_cat_2_cat_target_{}'.format(feature))


    #Drop the dummy_count column from data
    data.drop(['dummy_count'], axis=1, inplace = True)


def class_in_cat_feature(data=None, cat_features=None, plot=False, save_fig=False):
    '''
    Prints the categories and counts of a categorical feature

    Parameters:
    
    data: Pandas DataFrame or Series
    cat_features: Scalar, array, or list. 
               The categorical features in the dataset, if not provided, 
               we try to infer the categorical columns from the dataframe.
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if cat_features is None:
        cat_features = []
        for col in data.columns:
            if data[col].dtypes == 'object':
                cat_features.append(col)
                        

    for feature in cat_features:
        print('Class Count for', feature)
        print(data[feature].value_counts())
        print("-----------------------------")

    if plot:
        bar_cat_features(data, cat_features, save_fig=save_fig)

