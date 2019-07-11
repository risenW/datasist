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



def bar_cat_feats(data, cat_feats, fig_size=(5,5), save_fig=False):
    '''
    Makes a bar plot of all categorical features to show their counts.
    
    Parameters
    ------------

    data : Pandas dataframe.
    cat_feats: Scalar, array, or list. 
               The categorical features in the dataset, if not provided, 
               we try to infer the categorical columns from the dataframe.
    fig_size: tuple
              The size of the figure object
    '''

    if cat_feats is None:
        #TODO: Get categorical features from data
        pass

    else:

        for feat in cat_feats:
            fig = plt.figure(figsize=fig_size)
            ax = fig.gca()

            #get the value count of the column
            v_count = data[feat].value_counts()
            v_count.plot.bar(ax = ax)
            ax.set_title("Bar plot for " + feat)

            if save_fig:
                plt.savefig('Barplot_{}'.format(feat))





def box_num_2_cat(data=None, num_feats=None, target=None, fig_size=(5,5), large_data=False, save_fig=False):
    '''
    Makes a box plot of all numerical features against a specified categorical target column.
 
    Parameters
    ------------

    data : Pandas dataframe.
    num_feats: Scalar, array, or list. 
               The numerical features in the dataset, if not provided, 
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


    if num_feats is None:
        #TODO: Get numerical features from data
        pass
    
    if large_data:
        #use advanced sns boxenplot
         for feat in num_feats:
            fig = plt.figure(figsize=fig_size)
            ax = fig.gca()

            sns.set_style("whitegrid")
            sns.boxenplot(target, feat, data=data, ax=ax)
            plt.xlabel(feat) # Set text for the x axis
            plt.ylabel(target)# Set text for y axis
            plt.title('Box plot of {} against {}'.format(feat, target))
            if save_fig:
                plt.savefig('fig_{}_vs_{}'.format(feat,target))
            plt.show()
    else:
         for feat in num_feats:
            fig = plt.figure(figsize=fig_size)
            ax = fig.gca()

            sns.set_style("whitegrid")
            sns.boxplot(target, feat, data=data, ax=ax)
            plt.xlabel(feat) # Set text for the x axis
            plt.ylabel(target)# Set text for y axis
            plt.title('Box plot of {} against {}'.format(feat, target))
            if save_fig:
                plt.savefig('fig_{}_vs_{}'.format(feat,target))
            plt.show()




def violin_num_2_cat(data=None, num_feats=None, target=None, fig_size=(5,5), save_fig=False):
    '''
    Makes a violin plot of all numerical features against a specified categorical target column.
 
    Parameters
    ------------

    data : Pandas dataframe.
    num_feats: Scalar, array, or list. 
               The numerical features in the dataset, if not provided, 
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


    if num_feats is None:
        #TODO: Get numerical features from data
        pass

    for feat in num_feats:
        fig = plt.figure(figsize=fig_size)
        ax = fig.gca()

        sns.set_style("whitegrid")
        sns.violin(target, feat, data=data, ax=ax)
        plt.xlabel(feat) # Set text for the x axis
        plt.ylabel(target)# Set text for y axis
        plt.title('Violin plot of {} against {}'.format(feat, target))
        if save_fig:
            #TODO Add function to save to a specified directory
            plt.savefig('fig_{}_vs_{}'.format(feat,target))
        plt.show()
