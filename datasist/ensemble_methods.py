'''
This module contains all functions relating to ensemble methods. Bagging, Boosting, and Stacking.
'''
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score


def bagging(array_of_classifier = None, X = None, y = None, seed = 1075, cv = 10, max_samples = 1.0, max_features = 10):

	'''
	Bagging is a method to help decrease the variance of the classifier and reduce overfitting, by resampling data from the training set with the same cardinality as the original set. The model created should be less overfitted than a single individual model. A high variance for a model is not good.
	
	Parameters:
	---------------
	array_of_classifier: array, list.
		
		This is a list of instantiated classifiers to compare.
		
	X: DataFrame, Series, array.
		
		Independent Variables or Features.
	
	y: DataFrame, Series, array.
	
		Target variable.
		
	seed: int.

		A Pseudo-random number generator value.
		
	max_samples: int, float (default = 1.0).
	
		The number of samples to draw from X to train each base estimator
	
	max_featurs: int, float.
	
		The number of features to draw from X to train each base estimator.
	
	'''
	


	seed = seed
	np.random.seed(seed)
	
	
	if array_of_classifier is None:
		raise ValueError('Expects an array of Instantiated classifiers, got None')
	
	if X is None:
		raise ValueError('Expects a DataFrame of Features, got none')
		
	if y is None:
		raise ValueError(Expects y, target variable, got none')
		
		
	for clf in array_of_classifier:
		base_scores = cross_val_score(clf, X, y, cv = cv, n_jobs = -1)
		bagging_clf = BaggingClassifier(clf, max_samples = max_samples, max_features = max_features, random_state = seed)
		bagging_scores = cross_val_score(bagging_clf, X, y, cv = cv, n_jobs = -1)
		
		
		print('..........................', clf.__class__.__name__, '..........................')
        print('Mean: {}, standard_dev:(+/-) {} ----------Base classifier'.format(base_scores.mean(), base_scores.std()))
        print('Mean: {}, standard_dev:(+/-) {} ----------Bagging equivalent'.format(bagging_scores.mean(), bagging_scores.std()))
        print('/n')
	