from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kFold = KFold(5, True, 42)
gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11, learning_rate = 0.04)

score = cross_val_score(estimator=gbt, X = X, y = y, scoring = 'accuracy', cv = kFold)
print(score)

##gbt.fit(X, y)

##err_train = np.mean(y_train != gbt.predict(X_train))
##err_cv = np.mean(y_cv != gbt.predict(X_cv))
##print(err_train, err_cv)
