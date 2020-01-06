import os
import argparse  
import json
import joblib
import pickle
from pathlib import Path
import logging


def start_project(project_name=None):
    '''
    Creates a standard data science project directory. This helps in
    easy team collaboration, rapid prototyping, easy reproducibility and fast iteration. 
    
    The directory structure is by no means a globally recognized standard, but was inspired by
    the folder structure created by the Azure team (https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/overview)
    and Edward Ma (https://makcedward.github.io/) of OOCL.
    
    ### PROJECT STRUCTURE:

            ├── data
            │   ├── processed
            │   └── raw
            ├── models
            ├── notebooks
            │   ├── eda
            │   └── modeling
            ├── scripts
            │   ├── modeling
            │   ├── preparation
            │   └── processing
            ├── test

            DETAILS:

            data: Stores data used for the experiments, including raw and intermediate processed data.
                processed: stores all processed data files after cleaning, analysis, feature creation etc.
                raw: Stores all raw data obtained from databases, file storages, etc.

            models: Stores trained binary model files. This are models saved after training and evaluation for later use.

            notebooks: stores jupyter notebooks for exploration, modeling, evaluation, etc.
                eda: Stores notebook of exploratory data analysis.
                modeling: Stores notebook for modeling and evaluation of different models.

            scripts: Stores all code scripts usually in Python/R format. This is usually refactored from the notebooks.
                modeling: Stores all scripts and code relating to model building, evaluation and saving.
                preparation: Stores all scripts used for data preparation and cleaning.
                processing: Stores all scripts used for reading in data from different sources like databases or file storage.

            test: Stores all test files for code in scripts.
    
    
    Parameters:
    -------------
    project_name: String, Filepath
            Name of filepath of the directory to initialize and create folders.
    
    Returns:
    -------------
    None
    
    '''
    if project_name is None:
        raise ValueError("project_name: Expecting a string or filepath, got 'None'")

    
    basepath = os.path.join(os.getcwd(), project_name)
    datapath = os.path.join(basepath, 'data')
    modelpath = os.path.join(basepath, 'models')
    outputpath = os.path.join(basepath, 'outputs')

    
    os.makedirs(datapath, exist_ok=True)
    os.makedirs(os.path.join(datapath, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(datapath, 'processed'), exist_ok=True)

    os.makedirs(os.path.join(basepath, 'notebooks'), exist_ok=True)
    os.makedirs(os.path.join(basepath, 'notebooks', 'eda'), exist_ok=True)
    os.makedirs(os.path.join(basepath, 'notebooks', 'modeling'), exist_ok=True)

    os.makedirs(os.path.join(basepath, 'scripts'), exist_ok=True)
    os.makedirs(os.path.join(basepath, 'scripts', 'modeling'), exist_ok=True)
    os.makedirs(os.path.join(basepath, 'scripts', 'preparation'), exist_ok=True)
    os.makedirs(os.path.join(basepath, 'scripts', 'processing'), exist_ok=True)

    os.makedirs(modelpath, exist_ok=True)
    os.makedirs(outputpath, exist_ok=True)

    os.makedirs(os.path.join(basepath, 'test'), exist_ok=True)
    
    desc = '''
    PROJECT STRUCTURE:

    ├── data
    │   ├── processed
    │   └── raw
    ├── models
    ├── notebooks
    │   ├── eda
    │   └── modeling
    ├── scripts
    │   ├── modeling
    │   ├── preparation
    │   └── processing
    ├── test

    DETAILS:

    data: Stores data used for the experiments, including raw and intermediate processed data.
        processed: stores all processed data files after cleaning, analysis, feature creation etc.
        raw: Stores all raw data obtained from databases, file storages, etc.

    models: Stores trained binary model files. This are models saved after training and evaluation for later use.

    notebooks: stores jupyter notebooks for exploration, modeling, evaluation, etc.
        eda: Stores notebook of exploratory data analysis.
        modeling: Stores notebook for modeling and evaluation of different models.

    scripts: Stores all code scripts usually in Python/R format. This is usually refactored from the notebooks.
        modeling: Stores all scripts and code relating to model building, evaluation and saving.
        preparation: Stores all scripts used for data preparation and cleaning.
        processing: Stores all scripts used for reading in data from different sources like databases or file storage.

    test: Stores all test files for code in scripts.

    ''' 
    #project configuration settings
    json_config = {"description": "This file holds all confguration settings for the current project",
                    "basepath": basepath,
                    "datapath" : datapath,
                    "outputpath": outputpath,
                    "modelpath": modelpath}
    
    #create a readme.txt file to explain the folder structure
    with open(os.path.join(basepath, "README.txt"), 'w') as readme:
        readme.write(desc)
    
    with open(os.path.join(basepath, "config.txt"), 'w') as configfile:
        json.dump(json_config, configfile)
    
    print("Project Initialized successfully in {}".format(basepath))
    print("Check folder description in ReadMe.txt")




def save_model(model, name='model', method='joblib'):
    '''
    Save a trained machine learning model in the models folder.
    Folders must be initialized using the datasist start_project function.
    Creates a folder models if datasist standard directory is not provided.

    Parameters:
    ------------
    model: binary file, Python object
        Trained model file to save in the models folder.

    name: string
        Name of the model to save it with.

    method: string
        Format to use in saving the model. It can be one of [joblib, pickle or keras].

    Returns:
    ---------
    None

    '''

    if model is None:
        raise ValueError("model: Expecting a binary model file, got 'None'")
   
    #get file path from config file
    #we assume that the user is saving the model from the models folder
    config = None

    try:
        homepath = _get_home_path(os.getcwd())
        config_path = os.path.join(homepath, 'config.txt')
        
        with open(config_path) as configfile:
            config = json.load(configfile)
        
        model_path = os.path.join(config['modelpath'], name)

        if method is "joblib":
            filename = model_path + '.jbl'
            joblib.dump(model, model_path)
            print("model saved in {}".format(filename))
        elif method is 'pickle':
            filename = model_path + '.pkl'
            pickle.dump(model, model_path)
            print("model saved in {}".format(filename))

        elif method is 'keras':
            filename = model_path + '.h5'
            model.save(filename)
            print("model saved in {}".format(filename))

        else:
            logging.error("{} not supported, specify one of [joblib, pickle, keras]".format(method))
            
    except FileNotFoundError as e:
        msg = "models folder does not exist. Saving model to the {} folder. It is recommended that you start your project using datasist's start_project function".format(name)
        logging.info(msg)

        if method is "joblib":
            filename = name + '.jbl'
            joblib.dump(model, filename)
            print("model saved in current working directory")
        elif method is 'pickle':
            filename = name + '.pkl'
            pickle.dump(model, filename)
            print("model saved in current working directory")

        elif method is 'keras':
            filename = name + '.h5'
            model.save(filename)
            print("model saved in current working directory")

        else:
            logging.error("{} not supported, specify one of [joblib, pickle, keras]".format(method))
            


def save_data(data, name='processed_data', method=None, loc='processed'):
    
    '''
    Saves data in the data folder. The data folder contains the processed and raw subfolders.

    The processed subfolder holds data that have been processed by some methods and can be used for later computation. Files like
    feature matrixes, clean data files etc.

    The raw subfolder contains data in the raw format. This can be in the form of sql tables, csv files raw texts etc.
    Folders must be initialized using the datasist start_project function.

    Parameters:
    ------------
    data: binary strings, CSV, txt
        Data to save in the specified folder

    name: string, Default proc_data
        Name of the data file to save.

    method: string, Default None
        Format to use in saving the data. It can be empty string, and we assume it is a 
        Pandas DataFrame, and we use the to_csv function, else we serialize with joblib.
    
    loc: string, Default processed.
        subfolder to save the data file to. Can be one of [processed, raw ]

    Returns:
    ---------
    None

    '''
    if data is None:
        raise ValueError("data: Expecting a dataset, got 'None'")

    if loc not in ['processed', 'raw']:
        raise ValueError("loc: location not found, expecting one of [processed , raw] got {}".format(loc))
    
    try:
        homepath = _get_home_path(os.getcwd())
        config_path = os.path.join(homepath, 'config.txt')
        
        with open(config_path) as configfile:
            config = json.load(configfile)
        
        data_path = os.path.join(config['datapath'], loc)


        if method is "joblib":
            filename =  os.path.join(data_path, name) + '.jbl'
            joblib.dump(data, filename)
            print("Data saved in {}".format(filename))

        else:
            try:
                data.to_csv(os.path.join(data_path, name) + '.csv', index=False)
                print("Data saved successfully")

            except AttributeError as e:
                print("The file to save must be a Pandas DataFrame. Otherwise, change method parameter to joblib ")
                logging.error(e)                     


    except FileNotFoundError as e:
        msg = "data folder does not exist. Saving data to the {} folder. It is recommended that you start your project using datasist's start_project function".format(name)
        logging.info(msg)

        if method is "joblib":
            filename = name + '.jbl'
            joblib.dump(data, filename)
            print("data saved in current working directory")
        else:
            try:
                data.to_csv(name + '.csv',  index=False)
            except AttributeError as e:
                logging.info("The file to save must be a Pandas DataFrame, else change method to joblib ")
                logging.error(e)                 




def save_outputs(data=None, name='proc_outputs', method='joblib'):

    '''
    Saves files like vocabulary, class labels, mappings, encodings, images etc. in the outputs folder. 
    
    Parameters:
    ------------
    data: binary strings, CSV, txt
        Data to save in the folder

    name: string, Default proc_outputs
        Name of the data file to save.

    method: string, Default joblib
        Format to use in saving the data. It can be one of [csv, joblib, pickle].

    Returns:
    ---------
    None

    '''
    if data is None:
        raise ValueError("data: Expecting a dataset, got 'None'")

    if method not in ['csv', 'joblib', 'pickle']:
        raise ValueError("method: Expecting one of ['csv', 'joblib', 'pickle'] got {}".format(method))
    
    try:
        homepath = _get_home_path(os.getcwd())
        config_path = os.path.join(homepath, 'config.txt')
        
        with open(config_path) as configfile:
            config = json.load(configfile)
        
        outputs_path = config['outputpath']


        if method is "joblib":
            filename =  os.path.join(outputs_path, name) + '.jbl'
            joblib.dump(data, filename)
            print("Data saved in {}".format(filename))
        elif method is 'pickle':
            filename =  os.path.join(outputs_path, name) + '.pkl'
            pickle.dump(data, filename)
            print("Data saved in {}".format(filename))
        elif method is 'csv':
            data.to_csv(os.path.join(outputs_path, name) + '.csv', index=False)
            print("Data saved successfully")
        else:
            logging.error("An error occured while savng the file")                    


    except FileNotFoundError as e:
        msg = "outputs folder does not exist. Saving data to the current folder. It is recommended that you start your project using datasist's start_project function"
        logging.info(msg)

        if method is "joblib":
            filename =  name + '.jbl'
            joblib.dump(data, filename)
            print("Data saved in {}".format(filename))
        elif method is 'pickle':
            filename =  name + '.pkl'
            pickle.dump(data, filename)
            print("Data saved in {}".format(filename))
        elif method is 'csv':
            data.to_csv(name + '.csv', index=False)
            print("Data saved successfully")
        else:
            logging.error("An error occured while savng the file")                              



def _get_home_path(filepath):
    if filepath.endswith('scripts/modeling'):
        indx = filepath.index("scripts/modeling")
        path = filepath[0:indx]
        return path
    elif filepath.endswith('notebooks/modeling'):
        indx = filepath.index("notebooks/modeling")
        path = filepath[0:indx]
        return path
    elif filepath.endswith('notebooks/eda'):
        indx = filepath.index("notebooks/eda")
        path = filepath[0:indx]
        return path
    else:
        return filepath


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-p", "--project_name", help="Name of the parent directory to initialize folders")
#     args = parser.parse_args()

#     start_project(args.project_name)