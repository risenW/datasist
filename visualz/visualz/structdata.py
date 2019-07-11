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



def plot_bar(data, cat_feats, fig_size=(5,5)):
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

