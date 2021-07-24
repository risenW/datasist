import pytest
import pandas as pd
import numpy as np
from datasist import structdata
from datasist import feature_engineering

df1 = pd.DataFrame({'Name': ['tom', 'nick', 'jack', 'remi'],
                    'Age': ['20', '21', '19', '22'],
                    'Sex': ['Male', 'Male', 'Female', 'Female'],
                    'Date Resumed': ['10-11-19', '03-04-02', '09-04-15', '11-11-19'],
                    'Pension Date': ['01-02-2010', '06-09-2020', '09-04-2015', '10-02-2019']
                    })

def test_get_date_cols():
    data = df1
    expected = {'Date Resumed', 'Pension Date'}
    output = structdata.get_date_cols(feature_engineering.to_date(data))
    assert expected == output
