>>> import os
>>> os.chdir("D:\OneDrive\ML\py.w5.RF")
>>> import pandas
>>> data = pandas.read_csv("abalone.csv")
>>> import numpy as np

>>> data['Sex']=data['Sex'].map(lambda x:1 if x =='M' else (-1 if x == 'F' else 0))
>>> X = np.array(data[np.arange(0,8)])
>>> y = np.array(data['Rings'])

>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.model_selection import KFold
>>> kFold = KFold(5, True, 1)
>>> estimators_numbers = range(1, 51)

>>> from sklearn.metrics import r2_score
	
>>> for en in estimators_numbers:
	clf = RandomForestRegressor(n_estimators = en,random_state=1)	
	r2_ss = []
	for train_index, test_index in kFold.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		clf.fit(X_train, y_train)
		pred = clf.predict(X_test)
		r2_s = r2_score(y_test, pred)
		r2_ss.append(r2_s)
	r2_ss_mean = np.average(r2_ss)
	if r2_ss_mean > 0.52:
		res = en
		break
		
>>> res
22

>>> r2_ss_means = []
>>> for en in estimators_numbers:
	clf = RandomForestRegressor(n_estimators = en,random_state=1)	
	r2_ss = []
	for train_index, test_index in kFold.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		clf.fit(X_train, y_train)
		pred = clf.predict(X_test)
		r2_s = r2_score(y_test, pred)
		r2_ss.append(r2_s)
	r2_ss_mean = np.average(r2_ss)
	r2_ss_means.append(r2_ss_mean)
