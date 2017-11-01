def get_mean_roc_auc_score(kFold, X, y, clf):
    rass = []
    for train_index, test_index in kFold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        pred = clf.predict_proba(X_test)[:,1]
        ras = roc_auc_score(y_test,pred)
        rass.append(ras)	
		
    rass_mean = np.average(rass)
    return rass_mean

def fill_zeros(data):
    columns = data.columns
    rows_count = data.shape[0]
    empty_values_columns = []
    for c in columns:
        if (data[c].count() < rows_count):
            empty_values_columns.append(c)
            data[c].fillna(0, inplace=True)
    return empty_values_columns

def drop_cat_features(data):
    data = data.drop('lobby_type', 1)
    data = data.drop('r1_hero', 1)
    data = data.drop('r2_hero', 1)
    data = data.drop('r3_hero', 1)
    data = data.drop('r4_hero', 1)
    data = data.drop('r5_hero', 1)
    data = data.drop('d1_hero', 1)
    data = data.drop('d2_hero', 1)
    data = data.drop('d3_hero', 1)
    data = data.drop('d4_hero', 1)
    data = data.drop('d5_hero', 1)
    return data

def get_X_pick(data):
    X_pick = np.zeros((data.shape[0], 112))
    for i, match_id in enumerate(data.index):
        for p in range(5):
            X_pick[i, data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
    return X_pick


import pandas
data = pandas.read_csv('./features.csv', index_col='match_id')
data_test = pandas.read_csv('./features_test.csv', index_col='match_id')
data = data.drop('duration', 1)
data = data.drop('tower_status_radiant', 1)
data = data.drop('tower_status_dire', 1)
data = data.drop('barracks_status_radiant', 1)
data = data.drop('barracks_status_dire', 1)

empty_values_columns = fill_zeros(data)
print(empty_values_columns)

from sklearn.model_selection import KFold    
kFold = KFold(5, True, 42)

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators = 50, verbose = True,
                                 random_state = 241, learning_rate = 0.1)

import numpy as np
from sklearn.metrics import roc_auc_score 
y = np.array(data['radiant_win'])

X = np.array(data[np.arange(0,102)])

gbc_rass = []

import time
import datetime



start_time = datetime.datetime.now()		
gbc_rass_mean = get_mean_roc_auc_score(kFold, X, y, gbc)
print (gbc_rass_mean)
time = datetime.datetime.now() - start_time
print (time)
		
lr_rass = []

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_transform = scaler.transform(X)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2', random_state = 241, C = 1, verbose = True)		
lr_rass_mean = get_mean_roc_auc_score(kFold, X_transform, y, lr)
print (lr_rass_mean)

hero_types = []
hero_types.append(np.unique(data['r1_hero']))
hero_types.append(np.unique(data['r2_hero']))
hero_types.append(np.unique(data['r3_hero']))
hero_types.append(np.unique(data['r4_hero']))
hero_types.append(np.unique(data['r5_hero']))
hero_types.append(np.unique(data['d1_hero']))
hero_types.append(np.unique(data['d2_hero']))
hero_types.append(np.unique(data['d3_hero']))
hero_types.append(np.unique(data['d4_hero']))
hero_types.append(np.unique(data['d5_hero']))
hero_types = np.unique(hero_types)
N = hero_types.shape[0]
print (N)

X_pick = get_X_pick(data)

data = drop_cat_features(data)

X_no_cat = np.array(data[np.arange(0,91)])
scaler.fit(X_no_cat)
X_transform_no_cat = scaler.transform(X_no_cat)		
lr_rass_mean_no_cat = get_mean_roc_auc_score(kFold, X_transform_no_cat, y, lr)
print (lr_rass_mean_no_cat)

lr_rass_no_cat_pick = []
X_transform_no_cat_pick = np.concatenate([X_transform_no_cat, X_pick], axis=1)		
lr_rass_mean_no_cat_pick = get_mean_roc_auc_score(
    kFold, X_transform_no_cat_pick, y, lr)
print (lr_rass_mean_no_cat_pick)


fill_zeros(data_test)
X_test_pick = get_X_pick(data_test)
data_test = drop_cat_features(data_test)
X_test_no_cat = np.array(data_test[np.arange(0,91)])
X_test_transform_no_cat = scaler.transform(X_test_no_cat)
X_test_transform_no_cat_pick = np.concatenate([X_test_transform_no_cat, X_test_pick], axis=1)
radiant_pred = lr.predict_proba(X_test_transform_no_cat_pick)[:,1]
print (max(radiant_pred))
print (min(radiant_pred))

