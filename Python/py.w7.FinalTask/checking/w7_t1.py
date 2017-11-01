import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Gradient Boosting function
def calc_gbc(X_train,y_train,X_test,kf):
    print('-----------------\nGradient Boosting\n-----------------')
    # Checking different number of estimators
    '''
    for i in range(30,61,10):
        t1 = time.time()
        gbc = GradientBoostingClassifier(n_estimators=i, random_state=241)
        gbc.fit(X_train,y_train)
        score = np.mean(cross_val_score(gbc, X_train, y_train, cv=kf, scoring='roc_auc'))
        t2 = time.time()
        print('n_estimators: {:<3} Score: {:<7} Time: {:<7}'.format(i,score.round(4),round(t2-t1,4)))

    # Decreasing process time
    print('-----------------\nDecreasing Gradient Boosting time\n-----------------')
    
    # Decreasing max depth of trees
    for i in range(2,0,-1):
        t1 = time.time()
        gbc = GradientBoostingClassifier(n_estimators=30, max_depth=i)
        gbc.fit(X_train,y_train)
        score = np.mean(cross_val_score(gbc, X_train, y_train, cv=kf, scoring='roc_auc'))
        t2 = time.time()
        print('n_estimators: {:<3} Score: {:<7} Time: {:<7} Max depth: {:<3}'.format(30,score.round(4),round(t2-t1,4),gbc.max_depth))

    # Decreasing training set size
    X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_train, y_train, train_size=0.33, random_state=42)
    t1 = time.time()
    gbc = GradientBoostingClassifier(n_estimators=30)
    gbc.fit(X_train_reduced, y_train_reduced)
    score = np.mean(cross_val_score(gbc, X_test_reduced, y_test_reduced, cv=kf, scoring='roc_auc'))
    t2 = time.time()
    print('n_estimators: {:<3} Score: {:<7} Time: {:<7} Train size: {:<3}'.format(30,score.round(4),round(t2-t1,4),0.33))

    # Changing learning rate
    for l_r in [0.35,0.40,0.45,0.50,0.55,0.60]:
        t1 = time.time()
        gbc = GradientBoostingClassifier(n_estimators=30, learning_rate=l_r)
        gbc.fit(X_train, y_train)
        score = np.mean(cross_val_score(gbc, X_train, y_train, cv=kf, scoring='roc_auc'))
        t2 = time.time()
        print('n_estimators: {:<3} Score: {:<7} Time: {:<7} Learning rate: {:<3}'.format(30, score.round(4),
                                                                                     round(t2 - t1, 4), l_r))
    '''

    t1 = time.time()
    gbc = GradientBoostingClassifier(n_estimators=60, learning_rate=0.45, max_depth=2)
    X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_train, y_train, train_size=0.33, random_state=42)
    gbc.fit(X_train_reduced, y_train_reduced)
    score = np.mean(cross_val_score(gbc, X_train, y_train, cv=kf, scoring='roc_auc'))
    t2 = time.time()
    print('{} n_estimators: {:<3} Learning rate: {:<3} Max depth: {} Score: {:<7} Time: {:<7}'
          .format('Best Gradient Boosting',gbc.n_estimators,gbc.learning_rate,gbc.max_depth,score.round(4),round(t2 - t1, 4)))

def calc_lr(X_train,y_train,kf):
    print('-----------------\nLogistic Regression\n-----------------')
    X_heroes, X_without_heroes = make_heroes(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    dataset_names = iter(['Normal features set','Features set without heroes','Features set with heroes matrix'])
    for data in [X_train_scaled, X_without_heroes, X_heroes]:
        X_train_name = next(dataset_names)
        for c in [0.001,0.01,0.1,1,10]:
            t1 = time.time()
            lr = LogisticRegression(penalty = 'l2',C = c)
            lr.fit(X_train_scaled,y_train)
            score = np.mean(cross_val_score(lr, data, y_train, cv=kf, scoring='roc_auc'))
            t2 = time.time()
            print('{}  C: {:<3} Score: {:<7} Time: {:<7}'.format(X_train_name,c,round(score,4),round(t2-t1,4)))

def predict_test_results(X_train,y_train,X_test):
    print('===============')
    X_train_heroes, X_without_heroes = make_heroes(X_train)
    X_test_heroes, X_without_heroes = make_heroes(X_test)
    t1 = time.time()
    lr = LogisticRegression(penalty='l2', C=1)
    lr.fit(X_train_heroes,y_train)
    y_test_pred = lr.predict_proba(X_test_heroes)
    t2 = time.time()
    min_pred = 0
    min_dif = 0.5
    for i in y_test_pred[:, 1]:
        if abs(i-0.5)<min_dif:
            min_dif = abs(i-0.5)
            min_pred = i
    print('Max: ',max(y_test_pred[:, 1]))
    print('Min: ',1 - min_pred)

def make_heroes(data):
    scaler = StandardScaler()
    data_without_heroes = scaler.fit_transform(data.drop(axis=1,
                                                    labels=['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero',
                                                            'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero',
                                                            'd5_hero']))
    heroes_set = set()
    for i in range(1, 6):
        heroes_set = heroes_set.union(set(data['d{}_hero'.format(i)]))
        heroes_set = heroes_set.union(set(data['r{}_hero'.format(i)]))
    X_pick = np.zeros((data.shape[0], max(heroes_set)))
    for i, match_id in enumerate(data.index):
        for p in range(5):
            X_pick[i, data.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            X_pick[i, data.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
    X_heroes = pd.concat([pd.DataFrame(X_pick), pd.DataFrame(data_without_heroes)], axis=1)
    return X_heroes, data_without_heroes


if __name__ == "__main__":
    kf = KFold(n_splits=5, shuffle=True)
    X_train = pd.read_csv('./features.csv', index_col='match_id') \
        .drop(
        labels=['duration', 'radiant_win', 'barracks_status_dire', 'barracks_status_radiant', 'tower_status_radiant',
                'tower_status_dire'], axis=1).fillna(value = 0)
    y_train = pd.read_csv('./features.csv', index_col='match_id')['radiant_win']
    X_test = pd.read_csv('./features_test.csv', index_col='match_id').fillna(value=0)
    #calc_gbc(X_train,y_train,X_test,kf)
    calc_lr(X_train,y_train,kf)
    predict_test_results(X_train,y_train,X_test)