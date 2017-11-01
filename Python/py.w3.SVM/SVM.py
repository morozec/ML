>>> import os
>>> os.chdir("D:\OneDrive\py.w3.SVM")
>>> import numpy as np
>>> import pandas
>>> data = pandas.read_csv("svm-data.csv", header = None)
>>> y = np.array(data[0])
>>> X = np.array(data[[1,2]])
>>> from sklearn.svm import SVC
>>> svc = SVC(C=100000, kernel = 'linear', random_state=241)
>>> svc.fit(X,y)
SVC(C=100000, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=241, shrinking=True,
  tol=0.001, verbose=False)
>>> svc.support_


