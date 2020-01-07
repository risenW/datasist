'''
This module contains all functions relating to modeling in using sklearn library.

'''
import platform
import numpy as np
import pandas as pd
import seaborn as sns
from .visualizations import plot_auc
from sklearn.metrics import roc_curve, confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, make_scorer, mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
from sklearn.model_selection import KFold, cross_val_score


if platform.system() == "Darwin":
    import matplotlib as plt
    plt.use('TkAgg')
else:
    import matplotlib.pyplot as plt


def train_classifier(X_train = None, y_train=None, X_val=None, y_val=None, estimator=None, cross_validate=False, cv=5, show_roc_plot=True, save_plot=False):
    '''
    Train a classification estimator and calculate numerous performance metric.

    Parameters:
    ----------------------------
        X_train: Array, DataFrame, Series.

            The feature set (x) to use in training an estimator in other predict the outcome (y).

        y_train: Series, 1-d array, list

            The ground truth value for the train dataset

        X_val: Array, DataFrame. Series.

            The feature set (x) to use in validating a trained estimator.

        y_val: Series, 1-d array, list

            The ground truth value for the validation dataset.

        estimator: sklearn estimator.

            Sklearn estimator that implements the fit and predict functions.

        cross_validate: Bool, default False

            Use a cross validation strategy or not.
        
        cv: Int, default 5

            Number of folds to use in cross validation.
        
        show_roc_curve: Bool, default True.

            Plot a ROC plot showing estimator performance.
        
        save_plot: Bool, default False.

            Save the plot as a png file.
    
'''
    if X_train is None:
        raise ValueError("X_train: Expecting a DataFrame/ numpy2d array, got 'None'")
    
    if y_train is None:
        raise ValueError("y_train: Expecting a Series/ numpy1D array, got 'None'")

    #initialize variables to hold calculations
    pred, acc, f1, precision, recall, confusion_mat = 0, 0, 0, 0, 0, None

    if cross_validate:
        dict_scorers  = {'acc' : accuracy_score,
                        'f1' : f1_score,
                        'precision': precision_score, 
                        'recall' : recall_score
                        }


        metric_names = ['Accuracy', 'F1_score', 'Precision', 'Recall']

        for metric_name, scorer in zip(metric_names, dict_scorers):
            cv_score = np.mean(cross_val_score(estimator, X_train, y_train, scoring=make_scorer(dict_scorers[scorer]),cv=cv))
            print("{} is {}".format(metric_name,  round(cv_score * 100, 4)))

        #TODO Add cross validation function for confusion matrix
  
    else:
        if X_val is None:
            raise ValueError("X_val: Expecting a DataFrame/ numpy array, got 'None'")
        
        if y_val is None:
            raise ValueError("y_val: Expecting a Series/ numpy1D array, got 'None'")

        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_val)
        get_classification_report(y_val, pred, show_roc_plot, save_plot)



def plot_feature_importance(estimator=None, col_names=None):
    '''
    Plots the feature importance from a trained scikit learn estimator
    as a bar chart.

    Parameters:
    -----------
        estimator: scikit  learn estimator.

            Model that has been fit and contains the feature_importance_ attribute.

        col_names: list

            The names of the columns. Must map unto feature importance array.

    Returns:
    --------
    
        Matplotlib figure showing feature importances
    '''
    if estimator is None:
        raise ValueError("estimator: Expecting an estimator that implements the fit api, got None")
    if col_names is None:
        raise ValueError("col_names: Expecting a list of column names, got 'None'")
    
    if len(col_names) != len(estimator.feature_importances_):
        raise ValueError("col_names: Lenght of col_names must match lenght of feature importances")

    imps = estimator.feature_importances_
    feats_imp = pd.DataFrame({"features": col_names, "importance": imps}).sort_values(by='importance', ascending=False)
    sns.barplot(x='features', y='importance', data=feats_imp)
    plt.xticks(rotation=90)
    plt.title("Feature importance plot")
    plt.show()


def make_submission_csv(estimator=None, X_train=None, y_train=None, test_data=None,
                  sample_submision_file=None, sub_col_name=None, name="final_submission"):
    '''
    Train a estimator and makes prediction with it on the final test set. Also
    returns a sample submission file for data science competitions
    
    Parameters:
    -----------------------
    estimator: Sklearn estimator.

        An sklearn estimator that implements the fit and predict functions.

    X_train: Array, DataFrame, Series.

        The feature set (x) to use in training an estimator to predict the outcome (y).

    y_train: Series, 1-d array, list

        The ground truth value for the train dataset

    test_data: Array, DataFrame. Series.

        The final test set (x) to predict on.
    
    sample_submision_file: DataFrame, Series.

        A sample csv file provided by data science competition hosts to use in creating your final submission.

    sub_col_name: String.

        Name of the prediction column in the sample submission file.

    name: String.

        Name of the created submission file.

    Return:

        Csv file saved in current working directory
    '''
    estimator.fit(X_train, y_train)
    pred = estimator.predict(test_data)

    sub = sample_submision_file
    sub[sub_col_name] = pred
    sub.to_csv(name + '.csv', index=False)
    print("File has been saved to current working directory")



def get_classification_report(y_train=None, prediction=None, show_roc_plot=True, save_plot=False):
    '''
    Generates performance report for a classification problem.

    Parameters:
    ------------------
    y_train: Array, series, list.

        The truth/ground value from the train data set.
    
    prediction: Array, series, list.

        The predicted value by a trained model.

    show_roc_plot: Bool, default True.

        Show the model ROC curve.

    save_plot: Bool, default True.

        Save the plot to the current working directory.

    '''
    acc = accuracy_score(y_train, prediction)
    f1 = f1_score(y_train, prediction)
    precision = precision_score(y_train, prediction)
    recall = recall_score(y_train, prediction)
    confusion_mat = confusion_matrix(y_train, prediction)

    print("Accuracy is ", round(acc * 100))
    print("F1 score is ", round(f1 * 100))
    print("Precision is ", round(precision * 100))
    print("Recall is ", round(recall * 100))
    print("*" * 100)
    print("confusion Matrix")
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % confusion_mat[0,0] + '             %5d' % confusion_mat[0,1])
    print('Actual negative    %6d' % confusion_mat[1,0] + '             %5d' % confusion_mat[1,1])
    print('')

    if show_roc_plot:        
        plot_auc(y_train, prediction)

        if save_plot:
            plt.savefig("roc_plot.png")


def get_regression_report(y_true=None, prediction=None, show_r2_plot=True, save_plot=False):
    '''
    Generates performance report for a regression problem.

    Parameters:
    ------------------
    y_true: Array, series, list.

        The truth/ground value from the train data set.
    
    prediction: Array, series, list.

        The predicted value by a trained model.

    show_r2_plot: Bool, default True.

        Show the r-squared curve.

    save_plot: Bool, default True.

        Save the plot to the current working directory.

    '''
    mae = mean_absolute_error(y_true, prediction)
    mse = mean_squared_error(y_true, prediction)
    msle = precision_score(y_true, prediction)
    r2 = r2_score(y_true, prediction)
    
    print("Mean Absolute Error: ", round(mae, 5))
    print("Mean Squared Error: ", round(mse, 5))
    print("Mean Squared Log Error: ", round(msle, 5))
    print("R-squared Error:  ", round(r2, 5))
    print("*" * 100)

    if show_r2_plot:              
        plt.scatter(y_true,prediction)
        plt.xlabel('Truth values')
        plt.ylabel('Predicted values')
        plt.plot(np.unique(y_true), np.poly1d(np.polyfit(y_true, y_true, 1))(np.unique(y_true)))
        plt.text(0.7, 0.2, 'R-squared = %0.2f' % r2)
        plt.show()

        if save_plot:
            plt.savefig("r2_plot.png")





def compare_model(models_list=None, x_train=None, y_train=None, scoring_metric=None, scoring_cv=3, silenced=True, plot=True):
    """
    Train multiple user-defined model and display report based on defined metric. Enables user to pick the best base model for a problem.

    Parameters
    ----------------
        models_list: list

            a list of models to be trained

        x_train: Array, DataFrame, Series

            The feature set (x) to use in training an estimator to predict the outcome (y).

        y_train: Series, 1-d array, list

            The ground truth value for the train dataset

        scoring_metric: str

            Metric to use in scoring the model

        scoring_cv: int

            default value is 3

    Returns
    ---------------
    a tuple of fitted_model and the model evaluation scores
    """
    # if not _same_model:
    #     raise ValueError("model_list: models must be of the same class. Cannot use a classifier and a regressor.")


    if models_list is None or len(models_list) < 1:
        raise ValueError("model_list: model_list can't be 'None' or empty")

    if x_train is None:
        raise ValueError("x_train: features can't be 'None' or empty")

    if y_train is None:
        raise ValueError("y_train: features can't be 'None' or empty")



    fitted_model = []
    model_scores = []
    model_names = []

    for i, model in enumerate(models_list):
        if silenced is not True:
            print("Fitting {}".format(type(model).__name__))
        model.fit(x_train, y_train)
        # append fitted model into list
        fitted_model.append(model)
        model_score = np.mean(cross_val_score(model, x_train, y_train, scoring=scoring_metric, cv=scoring_cv))
        model_scores.append(model_score)
        model_names.append(type(fitted_model[i]).__name__)
            
    if plot:
        sns.pointplot(y=model_scores, x=model_names)
        plt.xticks(rotation=90)
        plt.title("Model comparison plot")
        plt.show()

    return fitted_model, model_scores

