
import pytest
import pandas as pd
import numpy as np
from datasist import timeseries


def test_time_elapsed():
    date1 = ['2019-07-27 19:00', '2019-07-27 18:00', '2019-07-27 15:00']
    date2 = ['2019-07-27 20:00', '2019-07-27 20:00', '2019-07-27 18:00']

    expected = pd.DataFrame([3600.0, 7200.0, 10800.0]).values
    output = timeseries.time_elapsed(date_cols=[date2, date1]).values
    np.testing.assert_array_equal(expected,output)
