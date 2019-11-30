'''
This module contains all functions relating to visualization.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from . import structdata
from IPython.display import display
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import sklearn.metrics as sklm



def countplot(data=None, features=None, separate_by=None, fig_size=(5,5), save_fig=False):
    '''
    Makes a bar plot of all categorical features to show their counts.
    
    Parameters
    ------------

        data : DataFrame, array, or list of arrays.

            The data to plot.

        features: str, scalar, array, or list. 

            The categorical features in the dataset, if not provided, 
            we try to infer the categorical columns from the dataframe.

        separate_by: str, default None.

            The feature used to seperate the plot. Called hue in seaborn.

        fig_size: tuple, Default (5,5)

            The size of the figure object.

        save_fig: bool, Default False.

            Saves the plot to the current working directory

    Returns
    -------
        None
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if features is None:
        features = structdata.get_cat_feats(data)
        
    for feature in features:
        #Check the size of categories in the feature: Anything greater than 20 is not plotted
        if len(data[feature].unique()) > 30:
            print("Unique Values in {} is too large to plot".format(feature))
            print('\n')
        else:
            fig = plt.figure(figsize=fig_size)
            ax = fig.gca()

            sns.countplot(x=feature, hue=separate_by, data=data)
            plt.xticks(rotation=90)
            ax.set_title("Count plot for " + feature)

            if save_fig:
                plt.savefig('Countplot_{}'.format(feature))




def plot_missing(data=None):
    '''
    Plots the data as a heatmap to show missing values

    Parameters
    ----------
        data: DataFrame, array, or list of arrays.
            The data to plot.
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    sns.heatmap(data.isnull(), cbar=True)
    plt.show()



def boxplot(data=None, num_features=None, target=None, fig_size=(5,5), large_data=False, save_fig=False):
    '''
    Makes a box plot of all numerical features against a specified categorical target column.

    A box plot (or box-and-whisker plot) shows the distribution of quantitative
    data in a way that facilitates comparisons between variables or across
    levels of a categorical variable. The box shows the quartiles of the
    dataset while the whiskers extend to show the rest of the distribution,
    except for points that are determined to be "outliers" using a method
    that is a function of the inter-quartile range

    Parameters
    ------------

        data : DataFrame, array, or list of arrays.

            Dataset for plotting.

        num_features: Scalar, array, or list. 

            The numerical features in the dataset, if not None, 
            we try to infer the numerical columns from the dataframe.

        target: array, pandas series, list.

            A categorical target column. Maximun number of categories is 10 and minimum is 1

        fig_size: tuple, Default (8,8)

            The size of the figure object.

        large_data: bool, Default False.

            If True, then sns boxenplot is used instead of normal boxplot. Boxenplot is 
            better for large dataset.

        save_fig: bool, Default False.

            If True, saves the current plot to the current working directory
        '''


    if target is None:
        raise ValueError('Target value cannot be None')

    if len(data[target].unique()) > 10:
        raise AttributeError("Target categories must be less than 10")

    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if num_features is None:
        num_features = structdata.get_num_feats(data)
    
    if large_data:
        #use advanced sns boxenplot
        for feature in num_features:
            fig = plt.figure(figsize=fig_size)
            ax = fig.gca()

            sns.set_style("whitegrid")
            sns.boxenplot(target, feature, data=data, ax=ax)
            plt.ylabel(feature) # Set text for the x axis
            plt.xlabel(target)# Set text for y axis
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
            plt.ylabel(feature) # Set text for the x axis
            plt.xlabel(target)# Set text for y axis
            plt.xticks(rotation=90)
            plt.title("Box plot of '{}' vs. '{}'".format(feature, target))
            if save_fig:
                plt.savefig('fig_{}_vs_{}'.format(feature,target))
            plt.show()




def violinplot(data=None, num_features=None, target=None, fig_size=(5,5), save_fig=False):
    '''
    Makes a violin plot of all numerical features against a specified categorical target column.

    A violin plot plays a similar role as a box and whisker plot. It shows the
    distribution of quantitative data across several levels of one (or more)
    categorical variables such that those distributions can be compared. Unlike
    a box plot, in which all of the plot components correspond to actual
    datapoints, the violin plot features a kernel density estimation of the
    underlying distribution.
    Parameters
    ------------

        data : DataFrame, array, or list of arrays.

            Dataset for plotting.

        num_features: Scalar, array, or list. 

            The numerical features in the dataset, if not None, 
            we try to infer the numerical columns from the dataframe.

        target: array, pandas series, list.

            A categorical target column. Maximun number of categories is 10 and minimum is 1.

        fig_size: tuple, Default (8,8)

            The size of the figure object.

        save_fig: bool, Default False.

            If True, saves the current plot to the current working directory
   '''

    if target is None:
        raise ValueError('Target value cannot be None')

    if len(data[target].unique()) > 10:
        raise AttributeError("Target categories must be less than 10")


    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if num_features is None:
        num_features = structdata.get_num_feats(data)

    for feature in num_features:
        fig = plt.figure(figsize=fig_size)
        ax = fig.gca()

        sns.set_style("whitegrid")
        sns.violinplot(target, feature, data=data, ax=ax)
        plt.xticks(rotation=90)
        plt.ylabel(feature) # Set text for the x axis
        plt.xlabel(target)# Set text for y axis
        plt.title("Violin plot of '{}' vs. '{}'".format(feature, target))
        if save_fig:
            #TODO Add function to save to a specified directory
            plt.savefig('fig_{}_vs_{}'.format(feature,target))
        plt.show()




def histogram(data=None, num_features=None, bins=None, show_dist_type=False, fig_size=(5,5), save_fig=False):
    '''
    Makes an histogram plot of all numerical features.
    Helps to show univariate distribution of the features.
    
    Parameters
    ------------
        data : DataFrame, array, or list of arrays.

            Dataset for plotting.

        num_features: Scalar, array, or list. 

            The numerical features in the dataset, if not None, 
            we try to infer the numerical columns from the dataframe.

        bins: int

            The number of bins to use.

        show_dist_type: bool, Default False

            If True, Calculates the skewness of the data and display one of (Left skewed, right skewed or normal) 

        fig_size: tuple, Default (8,8).

            The size of the figure object.

        save_fig: bool, Default False.

            If True, saves the current plot to the current working directory
    
    '''


    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if num_features is None:
        num_features = structdata.get_num_feats(data)

    for feature in num_features:
        fig = plt.figure(figsize=fig_size)
        ax = fig.gca()

        sns.distplot(data[feature].values, ax=ax, bins=bins)
        ax.set_xlabel(feature) # Set text for the x axis
        ax.set_ylabel('Count')# Set text for y axis

        if show_dist_type:
            ##TODO Add Code to calculate skewness
            pass
        else:
            ax.set_title('Histogram of ' + feature)

        if save_fig:
            #TODO Add function to save to a user specified directory
            plt.savefig('fig_hist_{}'.format(feature))

        plt.show()



def catbox(data=None, cat_features=None, target=None, fig_size=(10,5), save_fig=False):
    '''
    Makes a side by side bar plot of all categorical features against a categorical target feature.

    Parameters
    ------------

        data: DataFrame, array, or list of arrays.

            Dataset for plotting.

        cat_features: Scalar, array, or list. 

            The categorical features in the dataset, if None, 
            we try to infer the categorical columns from the dataframe.

        target: Scalar, array or list.

            Categorical target to plot against.

        fig_size: tuple, Default (12,6)

            The size of the figure object.

        save_fig: bool, Default False.

            If True, saves the plot to the current working directory.
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if cat_features is None:
        cat_features = structdata.get_cat_feats(data)

    #remove target from cat_features
    try:
        cat_features.remove(target)
    except:
        pass
    
    if len(data[target].unique()) > 8:
        #TODO Plot only a subset of the features say top 10
        raise AttributeError("Target categories must be less than seven")

    #Create a dummy column to hold count of values
    data['dummy_count'] = np.ones(shape = data.shape[0])
    #Loop over each categorical feature and plot the rate for each category.
    for feature in cat_features:
        #Plots are made for only categories with less than 15 unique values because of speed
        if len(data[feature].unique()) > 15 :
            print("{} feature has too many categories and will not be ploted".format(feature))
            
        else:     
            counts = data[['dummy_count', target, feature]].groupby([target, feature], as_index = False).count()
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
                plt.tight_layout(2)

                if save_fig:
                    plt.savefig('fig_catbox_{}'.format(feature))


    #Drop the dummy_count column from data
    data.drop(['dummy_count'], axis=1, inplace = True)


def class_count(data=None, features=None, plot=False, save_fig=False):
    '''
    Displays the number of classes in a categorical feature.

    Parameters:
    
        data: Pandas DataFrame or Series

            Dataset for plotting.

        features: Scalar, array, or list. 

            The categorical features in the dataset, if None, 
            we try to infer the categorical columns from the dataframe.

        plot: bool, Default False.

            Plots the class counts as a barplot

        save_fig: bool, Default False.

            Saves the plot to the current working directory.
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if features is None:
        features = structdata.get_cat_feats(data)

                        

    for feature in features:
        if data[feature].nunique() > 15:
            print("Unique classes in {} too large".format(feature))
        else:
            print('Class Count for', feature)
            display(pd.DataFrame(data[feature].value_counts()))

    if plot:
        countplot(data, features, save_fig=save_fig)



def scatterplot(data=None, num_features=None, target=None, separate_by=None, fig_size=(5,5), save_fig=False):
    '''
    Makes a scatter plot of numerical features against a numerical target.
    Helps to show the relationship between features.

    Parameters
    ------------
    
        data : DataFrame, array, or list of arrays.

            The data to plot.

        num_features: int/floats, scalar, array, or list. 

            The numeric features in the dataset, if not provided, 
            we try to infer the numeric columns from the dataframe.

        target: int/float, scalar, array or list.

            Numerical target feature to plot against.

        separate_by: str, default None.

            The feature used to seperate the plot. Called hue in seaborn.

        fig_size: tuple, Default (10,10)

            The size of the figure object.

        save_fig: bool, Default False.

            Saves the plot to the current working directory
        
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if separate_by is None:
        pass
    elif separate_by not in data.columns:
            raise ValueError("{} not found in data columns".format(separate_by))

    
    if target is None:
        raise ValueError('Target value cannot be None')

    if num_features is None:
        num_features = structdata.get_num_feats(data)

    for feature in num_features:
        fig = plt.figure(figsize=fig_size) # define plot area
        ax = fig.gca() # define axis  
        sns.scatterplot(x=feature, y=target, data=data, hue=separate_by)
        ax.set_title("Scatter Plot of '{}' vs. '{}' \n Separated by: '{}'".format(feature, target, separate_by))
        if save_fig:
            plt.savefig('fig_scatterplot_{}'.format(feature))




def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Parameters: 


    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax




def plot_auc(labels, predictions):
    '''
    Computes and plot the false positive rate, true positive rate and threshold along with the AUC
    Parameters:
    --------------------

    labels: 
        This is the true value ( in the case of binary either 0 or 1)

    predictions: 

        This is the probability that shows the likelihood of a value being 0 or 1

    Return:

        plots the Receiver operating characteristics.

    '''

    fpr, tpr, threshold = sklm.roc_curve(labels, predictions)
    auc = sklm.auc(fpr, tpr)

    ## Plot the result
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()




def plot_scatter_shape(data = None, cols = None, shape_col = '', col_y = '', alpha = 0.2):
    '''
    Makes a scatter plot of data using shape_col as seperation.
    Parameter:

        data: Dataframe 
            The data that is being imported using pandas.

        cols: list 
            The chosen number of columns in the DataFrame.

        shape_col: 
            The categorical column you want it to show as legend.

        col_y: The y axis of the plot
    
    Return:

        Matplotlib figure
    '''
    # pick distinctive shapes
    shapes = ['+', 'o', 's', 'x', '^']
    unique_cats = data[shape_col].unique()
    # loop over the columns to plot
    for col in cols:
        sns.set_style("whitegrid")
        # loop over the unique categories
        for i, cat in enumerate(unique_cats):
            temp = data[data[shape_col] == cat]
            sns.regplot(col, col_y, data=temp, marker = shapes[i], label = cat,
                        scatter_kws={"alpha":alpha}, fit_reg = False, color = 'blue')
        # Give the plot a main title
        plt.title('Scatter plot of ' + col_y + ' vs. ' + col)
        # Set text for the x axis
        plt.xlabel(col)
        # Set text for y axis
        plt.ylabel(col_y)
        plt.legend()
        plt.show()



def autoviz(data):
    '''
    Automatically visualize a data set. If dataset is large, autoViz uses a statistically valid sample for plotting.
    Parameter:
    --------------------
        data: Dataframe 
            The data to plot
            
    Return:
        Matplotlib figure
    '''
    #First check if autoviz is installed, if not installed, prompt the user to install it.
    import importlib.util
    import logging
    logging.basicConfig()

    package_name = 'autoviz'
    err_msg = "is not installed, to use this function, you must install " + package_name + ". \n To install, use 'pip install autoviz'"
    package_stat = importlib.util.find_spec(package_name)

    if package_stat is None:
        logging.error(package_name + " " + err_msg)
    else:
        from autoviz.AutoViz_Class import AutoViz_Class

        av = AutoViz_Class()
        av.AutoViz(filename='', dfte=data, max_cols_analyzed=50)