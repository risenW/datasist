
import pandas as pd
import numpy as np


def create_balanced_data(data, target_name, target_cats=None, n_classes=None, replacement=False ):
    '''
    Creates a balanced data set for training or testing from an imbalanced data set
    Parameter:
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame/ numpy2d array, got 'None'")
    
    if target_name is None:
        raise ValueError("target: Expecting a Series/ numpy1D array, got 'None'")
    
    temp_data = data.copy()
    new_data_size = sum(n_classes)
    classes = []
    class_index = []
    
    #get classes from data
    for t_cat in target_cats: 
        classes.append(temp_data[temp_data[target_name] == t_cat])
    
    for n_class, clas in zip(n_classes, classes):
        class_index.append(clas.sample(n_class, replace=True).index)
        
    #concat data together
    new_data = pd.concat([temp_data.loc[indx] for indx in class_index], ignore_index=True).sample(new_data_size).reset_index(drop=True)
    new_data_target = new_data[target_name]
    #drop new data target
    new_data.drop(target_name, axis=1, inplace=True)
    
    if not replacement:
        for indx in class_index:
            temp_data.drop(indx, inplace=True)
            
    #drop target from data
    original_target = temp_data[target_name]
    temp_data.drop(target_name, axis=1, inplace=True)
    
    print("shape of data {}".format(temp_data.shape))
    print("shape of data target {}".format(original_target.shape))
    print("shape of created data {}".format(new_data.shape))
    print("shape of created data target {}".format(new_data_target.shape))
    
    return temp_data, new_data, original_target, new_data_target
    