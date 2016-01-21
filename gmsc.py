# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from numpy import genfromtxt, savetxt
import pandas as pd
from numpy import genfromtxt, savetxt	

def main():
	'''
	Move the data set (training set and test data) in the same directory as the prediction script.
	
	'''
	dataset = pd.read_csv('cs-training.csv')	

	target = dataset['SeriousDlqin2yrs']
	train = dataset[dataset.columns.values[2:]]
	target.fillna(target.mean(), inplace = True)
	train.fillna(train.mean(), inplace = True)	

	test = pd.read_csv('cs-test.csv')
	test.fillna(test.mean(), inplace = True)
	test.drop('SeriousDlqin2yrs', axis = 1, inplace = True)
	test.drop('Unnamed: 0', axis = 1, inplace = True)	

	print "dataset made using numpy function genfromtxt"	

	print 'training set size = ',len(train)	

	rf = RandomForestClassifier(n_estimators = 256, oob_score = True, n_jobs = -1, random_state = 42)	

	rf.fit(train, target)

	print 'Model Fitted\n Writing to File...'

	predictions = [[index + 1, x[1]] for index, x in enumerate(rf.predict_proba(test))]
	predicted_probs = pd.DataFrame(predictions, columns = ['Id', 'Probability'])

	predicted_probs.to_csv('submission.csv',  index = False)
	
	print ('Predictions Made, Expected Score = ' +  str(rf.oob_score_))

if __name__ == '__main__' : main()


