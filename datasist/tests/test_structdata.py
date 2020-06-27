import pytest
import pandas as pd
import numpy as np
from datasist import structdata

df = pd.DataFrame({'country': ['Nigeria', 'Ghana', 'USA', 'Germany'],
                    'size': [280, 20, 60, np.NaN],
                    'language': ['En', 'En', 'En', 'EN']})

def test_get_top_5():
    expected = ['Nigeria', 'Ghanan']
    result = list(structdata.get_top_5(df)['country'].values)
    assert expected == result