import pytest
import pandas as pd
import numpy as np
from datasist import feature_engineering

df = pd.DataFrame({'country': ['Nigeria', 'Ghana', 'USA', 'Germany'],
                    'size': [280, 20, 60, np.NaN],
                    'language': ['En', 'En', 'En', np.NaN]})

df2 = pd.DataFrame({'country': ['Nigeria', 'Ghana', 'USA', 'Germany'],
                    'size': [180, np.NaN, np.NaN, np.NaN]})

df_dist = pd.DataFrame({'lat1': [10, 11, 12, 14],
                    'lat2': [30, 20, 10, 20],
                    'long1': [1,2,3,4],
                    'long2': [2,4,6,8]})

#dataset to testing age bin
table = pd.DataFrame(np.random.randint(16, 70,size=(50,2)),  columns=('Age','just_number'))
table['Salary'] = np.random.randint(5000, 10000, size=(50,))
table['Time'] = np.round(np.random.uniform(0.30, 60, size=(50,1)), 2)


def test_drop_missing():
    expected = ['country']
    output = list(feature_engineering.drop_missing(df2, 75).columns)
    assert expected == output
    
def test_drop_missing_greater():
    expected = ['country', 'size']
    output = list(feature_engineering.drop_missing(df2, 90).columns)
    assert expected == output

def test_drop_redundant():
    expected = ['country', 'size']
    output = list(feature_engineering.drop_redundant(df).columns)
    assert expected == output

def test_fill_missing_cats():
    expected = ['En']
    output = feature_engineering.fill_missing_cats(df)['language']
    assert expected == list(output.unique())

def test_fill_missing_num_mean():
    expected = 120
    output = feature_engineering.fill_missing_num(df)['size']
    assert expected == output[3]

def test_harversine_distance():
    expected = [3253, 2029, 1093, 1722]
    output = list(feature_engineering.haversine_distance(df_dist['lat1'], df_dist['lat2'], df_dist['long1'], df_dist['long2']).astype('int64'))
    assert expected == output
    
def test_manhattan_distance():
    expected = [37, 25, 13, 22]
    output = list(feature_engineering.manhattan_distance(df_dist['lat1'], df_dist['lat2'], df_dist['long1'], df_dist['long2']).astype('int64'))
    assert expected == output

def test_bearing():
    expected = [-106, -118, -155, -129]
    output = list(feature_engineering.bearing(df_dist['lat1'], df_dist['lat2'], df_dist['long1'], df_dist['long2']).astype('int64'))
    assert expected == output

def test_get_location_center():
    expected = [20, 15, 11, 17]
    output = list(feature_engineering.get_location_center(df_dist['lat1'], df_dist['lat2']).astype('int64'))
    assert expected == output

def test_bin_age():
    data_bin = [10, 20, 30, 40, 60, 70]
    data_label = ['1st level', '2nd level', '3rd level', '5th level', '6th level']
    expected = 5
    temp_df = feature_engineering.bin_age(data = table, feature = ['Age'], bins = data_bin, fill_missing = 'mean',
                                            labels = data_label, drop_original= True)
    output = temp_df['Age_binned'].nunique()
    assert expected == output