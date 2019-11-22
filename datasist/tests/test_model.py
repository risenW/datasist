
import pytest

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier

from datasist import model

def test_compare_model():
    x_train, y_train = make_classification(
        n_samples=50, 
        n_features=4,
        n_informative=2, 
        n_redundant=0, 
        random_state=0,
        shuffle=False
    )
    model_list = [
        RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0),
        BaggingClassifier(),
        GradientBoostingClassifier()
    ]
    fitted_model, model_scores = model.compare_model(model_list, x_train, y_train, 'accuracy')
    assert type(fitted_model) is list
    assert type(model_scores) is list
    assert hasattr(fitted_model[0], "predict_prob")
