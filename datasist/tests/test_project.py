import os
import shutil
from datasist import project
from sklearn.ensemble import RandomForestClassifier
from datasist.project import save_model, save_data, save_outputs, _get_home_path, get_data, get_model, get_output
import json
import pandas as pd
import logging



#setup and teardown class to run before any test case
def setup_function(module):
    project.startproject("tests/sampletest")
    logging.info("starting")
 
def teardown_function(module):
    shutil.rmtree("tests/sampletest")
    logging.info("tearing")



def test_start_project():
    expected = ['README.txt',
                'src',
                'data',
                'config.txt',
                'outputs']
    project.startproject("tests/sampletest")
    output = os.listdir("tests/sampletest")
    assert set(expected) == set(output)
    assert len(expected) == len(output)


def test_save_model_joblib():
    rf = RandomForestClassifier()
    save_model(model=rf, name='tests/randomforest', method='jb')
    expected = 'randomforest.jbl'
    output = os.listdir('tests/')
    assert expected in output
    os.remove('tests/randomforest.jbl')



def test_save_data_csv(): #Test data saving in a directory structure created with datasist start_project function
    expected1 = 'proc_file.csv'
    expected2 = 'raw_file.csv'

    config_path = os.path.join('tests/sampletest', 'config.txt')
    with open(config_path) as configfile:
        config = json.load(configfile)
    
    data_path_raw = os.path.join(config['datapath'], 'raw')
    data_path_proc = os.path.join(config['datapath'], 'processed')

    aa = pd.DataFrame([1,2,3,4,5])

    save_data(aa,name=data_path_proc + '/proc_file', method='csv', loc='processed')
    save_data(aa,name=data_path_raw + '/raw_file',method='csv', loc='raw')

    assert expected1 in os.listdir(data_path_proc)
    assert expected2 in os.listdir(data_path_raw)


def test_save_data_jbl(): #Test data saving in a directory structure created with datasist start_project function
    expected1 = 'proc_file.jbl'
    expected2 = 'raw_file.jbl'
    aa = pd.DataFrame([1,2,3,4,5])

    config_path = os.path.join('tests/sampletest', 'config.txt')
    with open(config_path) as configfile:
        config = json.load(configfile)
    
    data_path_raw = os.path.join(config['datapath'], 'raw')
    data_path_proc = os.path.join(config['datapath'], 'processed')


    save_data(aa,name=data_path_proc + '/proc_file', method='jb',loc='processed')
    save_data(aa,name=data_path_raw + '/raw_file',method='jb', loc='raw')

    assert expected1 in os.listdir(data_path_proc)
    assert expected2 in os.listdir(data_path_raw)


def test_save_data_before_init(): #Test data saving in an un-initialized project
    expected1 = 'proc_file.jbl'
    expected2 = 'raw_file.jbl'
    
    aa = pd.DataFrame([1,2,3,4,5])

    save_data(aa, name='tests/proc_file', method='jb')
    save_data(aa, name='tests/raw_file', method='jb')

    assert expected1 in os.listdir('tests')
    assert expected2 in os.listdir('tests')
    os.remove('tests/proc_file.jbl')
    os.remove('tests/raw_file.jbl')



def test_save_outputs_csv(): #Test data saving in a directory structure created with datasist start_project function
    expected = 'proc_outputs.csv'

    config_path = os.path.join('tests/sampletest', 'config.txt')
    with open(config_path) as configfile:
        config = json.load(configfile)
    
    output_path= os.path.join(config['outputpath'])
    aa = pd.DataFrame([1,2,3,4,5])
    save_outputs(data=aa,name=output_path + '/proc_outputs', method='csv')
    assert expected in os.listdir(output_path)

def test_save_outputs_jbl(): #Test data saving in a directory structure created with datasist start_project function
    expected = 'proc_outputs.jbl'

    config_path = os.path.join('tests/sampletest', 'config.txt')
    with open(config_path) as configfile:
        config = json.load(configfile)
    
    output_path= os.path.join(config['outputpath'])
    aa = pd.DataFrame([1,2,3,4,5])
    save_outputs(data=aa,name=output_path + '/proc_outputs')
    assert expected in os.listdir(output_path)
    

def test_save_output_before_init(): #Test output saving in an un-initialized datasist project
    expected = 'out_file.csv'    
    aa = pd.DataFrame([1,2,3,4,5])
    save_outputs(aa, name='tests/out_file', method='csv')

    assert expected in os.listdir('tests')
    os.remove('tests/out_file.csv')


def test_get_data_jb():
    expected = [1,2,3,4]
    save_data(expected, name='tests/sampletest/data/processed/proce', method='jb')
    output = get_data(path='tests/sampletest/data/processed/proce.jbl')

    assert expected == output
    assert type(expected) == type(output)



def test_get_model():
    rf = RandomForestClassifier()
    save_model(rf, name='tests/sampletest/outputs/models/rf_model', method='jb')
    output = get_model(path='tests/sampletest/outputs/models/rf_model.jbl')
    assert hasattr(rf, 'fit')



def test_get_output():
    temp = pd.DataFrame([1,2,3,4,5,6])
    save_model(temp, name='tests/sampletest/outputs/submit', method='jb')
    output = get_output(path='tests/sampletest/outputs/submit.jbl')
    assert hasattr(temp,'sample')
