import os
import shutil
from datasist.project import start_project
from sklearn.ensemble import RandomForestClassifier
from datasist.project import save_model

def test_start_project():
    expected = ['notebooks',
                'README.txt',
                'data',
                'test',
                'scripts',
                'config.txt',
                'models',
                'outputs']
    start_project("tests/sampletest")
    output = os.listdir("tests/sampletest/")
    print(output)
    assert expected == output
    # clean directory
    shutil.rmtree("tests/sampletest")



def test_save_model():
    rf = RandomForestClassifier()
    save_model(model=rf, name='tests/randomforest', method='joblib')
    expected = 'randomforest.jbl'
    output = os.listdir('tests/')
    assert expected in output
    os.remove('tests/randomforest.jbl')
