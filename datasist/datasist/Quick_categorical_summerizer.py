'''
This module contains all functions relating to the cleaning and exploration of structured data sets; mostly in pandas format
'''


import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .visualizations import class_count, plot_missing
from IPython.display import display


def quick_CSummarizer(dataframe, x=None, y=None, hue=None, palette='Set1', verbose=True):
    '''
    Helper function that gives a quick summary of a given column of categorical data
    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data, y would be the count
    y: str. vertical axis to plot the labels of categorical data, x would be the count
    hue: str. if you want to compare it another variable (usually the target variable)
    palette: array-like. Colour of the plot
    Returns
    =======
    Quick Stats of the data and also the count plot
    '''
    if x == None:
        column_interested = y
    else:
        column_interested = x
    series = dataframe[column_interested]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette=palette)
    plt.show()
    
    
    '''
    E.g
    c_palette = ['tab:blue', 'tab:orange']
categorical_summarized(train, y = 'date_block_num', palette=c_palette)
    '''
