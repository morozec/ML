>>> import os
>>> import pandas
>>> import numpy as np
>>> os.chdir("D:\OneDrive\py.w2.Perceptron")
>>> data = pandas.read_csv("perceptron-train.csv", header = None)
>>> data_test = pandas.read_csv("perceptron-test.csv", header = None)
>>> y_train = np.array(data[0])
>>> X_train = np.array(data[[1,2]])
>>> y_test = np.array(data_test[0])
>>> X_test = np.array(data_test[[1,2]])
>>> from sklearn.linear_model import Perceptron
>>> clf = Perceptron(random_state=241)
>>> clf.fit(X_train, y_train)
Perceptron(alpha=0.0001, class_weight=None, eta0=1.0, fit_intercept=True,
      n_iter=5, n_jobs=1, penalty=None, random_state=241, shuffle=True,
      verbose=0, warm_start=False)
>>> predictions = clf.predict(X_test)
>>> from sklearn.metrics import accuracy_score
>>> ac_s = accuracy_score(y_test, predictions)
>>> ac_s
0.65500000000000003
>>> from sklearn.preprocessing import StandardScaler
>>> scaler = StandardScaler()
>>> X_train_scaled = scaler.fit_transform(X_train)
>>> X_test_scaled = scaler.transform(X_test)
>>> clf_scaled = Perceptron(random_state=241)
>>> clf_scaled.fit(X_train_scaled, y_train)
Perceptron(alpha=0.0001, class_weight=None, eta0=1.0, fit_intercept=True,
      n_iter=5, n_jobs=1, penalty=None, random_state=241, shuffle=True,
      verbose=0, warm_start=False)
>>> predictions_scaled = clf_scaled.predict(X_test_scaled)
>>> ac_s_scaled = accuracy_score(y_test, predictions_scaled)
>>> ac_s_scaled
0.84499999999999997
>>> diff= ac_s_scaled - ac_s
>>> diff
0.18999999999999995
>>> 
