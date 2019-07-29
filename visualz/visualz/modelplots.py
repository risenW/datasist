from sklearn.metrics import roc_curve,confusion_matrix, precision_score,accuracy_score, recall_score,f1_score, make_scorer
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def train_classifier(train_data = None, target=None, val_data=None, val_data_target=None, model=None, cross_validate=False, cv=5, show_roc_plot=True, save_plot=False):
    '''
    train a classification model and returns all the popular performance
    metric
    Parameters:
    #TODO update Doc
    
'''
    if train_data is None:
        raise ValueError("train_data: Expecting a DataFrame/ numpy2d array, got 'None'")
    
    if target is None:
        raise ValueError("target: Expecting a Series/ numpy1D array, got 'None'")

    #initialize variables to hold calculations
    pred, acc, f1, precision, recall, confusion_mat = 0, 0, 0, 0, 0, None

    if cross_validate:
        dict_scorers  = {'acc' : accuracy_score,
                    'f1' : f1_score,
                    'precision': precision_score, 
                    'recall' : recall_score}

        metric_names = ['Accuracy', 'F1_score', 'Precision', 'Recall']

        for metric_name, scorer in zip(metric_names, dict_scorers):
            cv_score = np.mean(cross_val_score(model, train_data, target, scoring=make_scorer(dict_scorers[scorer]),cv=cv))
            print("{} is {}".format(metric_name,  round(cv_score * 100, 4)))
        #TODO Add cross validation function for confusion matrix
  
    else:
        if val_data is None:
            raise ValueError("val_data: Expecting a DataFrame/ numpy2d array, got 'None'")
        
        if val_data_target is None:
            raise ValueError("val_data_target: Expecting a Series/ numpy1D array, got 'None'")

        model.fit(train_data, target)
        pred = model.predict(val_data)
        get_classification_report(val_data_target, pred, show_roc_plot, save_plot)


def train_predict(model=None, train_data=None, target=None, test_data=None, make_submission_file=False,
                  sample_submision_file=None, submission_col_name=None, 
                  submision_file_name=None):
    '''
    Train and model and makes prediction with it on the final test set. Also
    returns a sample submission file for data science competitions
    
    Parameters:

    '''
    model.fit(train_data, target)
    pred = model.predict(test_data)

    if make_submission_file:
        sub = sample_submision_file
        sub[submission_col_name] = pred
        sub.to_csv(submision_file_name + '.csv', index=False)
        print("File has been saved to current working directory")
    else:
        return pred


def get_classification_report(target=None, pred=None, show_roc_plot=True, save_plot=False):
    acc = accuracy_score(target, pred)
    f1 = f1_score(target, pred)
    precision = precision_score(target, pred)
    recall = recall_score(target, pred)
    confusion_mat = confusion_matrix(target, pred)

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
        fpr, tpr, thresholds = roc_curve(target, pred)
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        plt.plot(fpr, tpr, marker='.')
        # show the plot
        plt.title("roc curve")
        plt.show()

        if save_plot:
            plt.savefig("roc_plot.png")
