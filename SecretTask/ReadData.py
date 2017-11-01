import pandas
import numpy as np
import math
data = pandas.read_csv("x_train.csv", sep=';', header=None)


mean = data.mean()
std = data.std()
data_norm = (data - mean)/std
X = np.array(data_norm)

##removing_indexes = []
##app_max = mean + 3 * std
##app_min = mean - 3 * std
##
##for i in range(X.shape[0]):
##    for j in range(X.shape[1]):
##        if X[i][j] < app_min[j] or X[i][j] > app_max[j]:
##            X[i][j]=mean[j]
            


##log_indexes = []

##column_count = X.shape[1]
##for i in range(column_count):
##    column = X[:,i]
##    if min(column) > 0:
##        X[:,i] = np.log(X[:,i])
##        log_indexes.append(i)


data_y = pandas.read_csv("y_train.csv", sep=';', header=None)
y = np.array(data_y).ravel()

#X = np.delete(X, removing_indexes, axis=0)
#y = np.delete(y, removing_indexes, axis=0)

data_test = pandas.read_csv("x_test.csv", sep=';', header=None)
data_test_norm = (data_test - mean)/std
X_test = np.array(data_test_norm)
##for i in log_indexes:
##    X_test[:,i] = np.log(X_test[:,i])


from sklearn.model_selection import KFold
kFold = KFold(5, True, 1)

from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.3, random_state = 11)
