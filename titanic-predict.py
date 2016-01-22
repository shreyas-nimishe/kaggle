# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB


def main():

	dataset = pd.read_csv('train.csv')		
	dataset['Sex'].replace(['male', 'female'], [0, 1], inplace = True)

	numeric_attr =  dataset.dtypes[dataset.dtypes != 'object'].index
	dataset[numeric_attr].head(5)	

	for attr in numeric_attr:
	    dataset[attr].fillna(dataset[attr].mean(), inplace = True)

	X = dataset[[a for a in numeric_attr if a != 'Survived']]	# Only numeric quatities considered for initial prediction
	Y = dataset['Survived']	
	
	print '\n\n\tFitting Model...\n\n'

	rf = RandomForestClassifier(n_estimators = 2000, n_jobs = -1, oob_score = True, random_state = 42)
	rf.fit(X, Y)	

	test = pd.read_csv('test.csv')
	print 'Total Test Data: ', len(test), '\n'	

	print 'Features Considered: ', X.columns.values, '\n'
	test = test[X.columns.values]
	test['Sex'].replace(['male', 'female'], [0, 1], inplace = True)
	for attr in X.columns.values:
	    test[attr].fillna(test[attr].mean(), inplace = True)	

	test = test[X.columns.values]
	for attr in X.columns.values:
	    test[attr].fillna(test[attr].mean(), inplace = True)	

	prediction = rf.predict(test)	

	results = []
	index = 0
	for x in prediction:
	    results.append([index + 892, x])
	    index += 1	
	
	output_prediction = pd.DataFrame( results, columns = ["PassengerId","Survived"] ).to_csv('titanic_submission.csv', index = False)
	print 'Predictions Made, Score Expected = ', rf.oob_score_, '\n'

if __name__ == '__main__': main()
