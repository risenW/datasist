
import pytest
import os
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor
from datasist.model import compare_model

def test_compare_model_classification():
    x_train, y_train = make_classification(
        n_samples=50, 
        n_features=4,
        n_informative=2, 
        n_redundant=0, 
        random_state=0,
        shuffle=False
    )
    model_list = [
        RandomForestClassifier(n_estimators=5, max_depth=2, random_state=0),
        GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=0)
    ]
    fitted_model, model_scores = compare_model(model_list, x_train, y_train, 'accuracy', plot=False)
    assert type(fitted_model) is list
    assert type(model_scores) is list
    assert hasattr(fitted_model[0], "predict")


def test_compare_model_regression():
    x_train, y_train = make_classification(
        n_samples=50, 
        n_features=4,
        n_informative=2, 
        n_redundant=0, 
        random_state=0,
        shuffle=False
    )
    model_list = [
        RandomForestRegressor(n_estimators=5, max_depth=2, random_state=0),
        GradientBoostingRegressor(n_estimators=5, max_depth=2, random_state=0)
    ]
    fitted_model, model_scores = compare_model(model_list, x_train, y_train, 'neg_mean_absolute_error', plot=False)
    assert type(fitted_model) is list
    assert type(model_scores) is list
    assert hasattr(fitted_model[0], "predict")

