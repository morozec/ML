>>> import os
>>> os.chdir("D:\OneDrive\ML\py.w5.GB")
>>> import numpy as np
>>> import pandas
>>> data = pandas.read_csv("gbm-data.csv")
>>> y = data.values[:,0]
>>> X = data.values[:,range(1, 1777)]
>>> from sklearn.model_selection import train_test_split
>>> X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.8, random_state = 241)
>>> learning_rates = [1, 0.5, 0.3, 0.2, 0.1]
>>> from sklearn.ensemble import GradientBoostingClassifier
>>> from sklearn.metrics import log_loss

>>> import matplotlib.pyplot as plt

>>> gbc = GradientBoostingClassifier(n_estimators = 250, verbose = True, random_state = 241, learning_rate = 0.2)
>>> gbc.fit(X_train, y_train)
      Iter       Train Loss   Remaining Time 
         1           1.2613           11.21s
         2           1.1715           12.77s
         3           1.1009           13.09s
         4           1.0529           13.53s
         5           1.0130           13.18s
         6           0.9740           12.89s
         7           0.9475           12.25s
         8           0.9197           12.10s
         9           0.8979           11.78s
        10           0.8730           11.62s
        20           0.7207            9.67s
        30           0.6055            9.31s
        40           0.5244            8.70s
        50           0.4501            8.34s
        60           0.3908            7.79s
        70           0.3372            7.50s
        80           0.3009            6.97s
        90           0.2603            6.58s
       100           0.2327            6.12s
       200           0.0835            1.93s
GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.2, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=None,
              min_impurity_split=1e-07, min_samples_leaf=1,
              min_samples_split=2, min_weight_fraction_leaf=0.0,
              n_estimators=250, presort='auto', random_state=241,
              subsample=1.0, verbose=True, warm_start=False)
>>> test_loss=[]
>>> train_loss = []


>>> for i, y_pred in enumerate(gbc.staged_decision_function(X_test)):
	y_pred_arr = np.array(y_pred)
	y_pred_sigm = 1/(1+np.exp(-y_pred_arr))
	tl = log_loss(y_test, y_pred_sigm)
	test_loss.append(tl)

	
>>> for i, y_pred in enumerate(gbc.staged_decision_function(X_train)):
	y_pred_arr = np.array(y_pred)
	y_pred_sigm = 1/(1+np.exp(-y_pred_arr))
	tl = log_loss(y_train, y_pred_sigm)
	train_loss.append(tl)

	
>>> plt.figure()
<matplotlib.figure.Figure object at 0x00000000164B2908>
>>> plt.plot(test_loss, 'r', linewidth=2)
[<matplotlib.lines.Line2D object at 0x00000000167C5E80>]
>>> plt.plot(train_loss, 'g', linewidth=2)
[<matplotlib.lines.Line2D object at 0x00000000164AC048>]
>>> plt.legend(['test', 'train'])
<matplotlib.legend.Legend object at 0x00000000167D0940>
>>> plt.show()
>>> min(test_loss)
0.53145079631906378
>>> test_loss.index(min(test_loss))
36

>>> from sklearn.ensemble import RandomForestClassifier
>>> rfc = RandomForestClassifier(n_estimators = 36, random_state = 241)
>>> rfc.fit(X_train,y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=36, n_jobs=1, oob_score=False, random_state=241,
            verbose=0, warm_start=False)
>>> pred = rfc.predict_proba(X_test)
>>> ll = log_loss(y_test, pred)
>>> ll
0.54138128618040693
>>> rfc = RandomForestClassifier(n_estimators = 500, random_state = 241)
>>> rfc.fit(X_train,y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=500, n_jobs=1, oob_score=False, random_state=241,
            verbose=0, warm_start=False)
>>> pred = rfc.predict_proba(X_test)
>>> ll = log_loss(y_test, pred)
>>> ll
0.52391560287710337
