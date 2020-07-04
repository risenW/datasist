import pytest
import pandas as pd
import numpy as np
from datasist import structdata

df = pd.DataFrame({'country': ['Nigeria', 'Ghana', 'USA', 'Germany'],
                    'size': [280, 20, 60, np.NaN],
                    'language': ['En', 'En', 'En', np.NaN]})

def test_train_test_split():
    expected_train_df = pd.DataFrame({'country': ['Nigeria', 'Ghana'],
                    'size': [280, 20],
                    'language': ['En', 'En']})
    expected_test_df = pd.DataFrame({'country': ['USA', 'Germany'],
                    'size': [ 60, np.NaN],
                    'language': [ 'En', np.NaN]})           
    train_output, test_output = structdata.train_test_split(df,0.5)
    
    assert len(expected_train_df) == len(train_output)
    assert len(expected_test_df)  == len(test_output)