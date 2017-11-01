from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

C_array = np.logspace(-3, 3, num=7)
svc = SVC(kernel='linear')
grid = GridSearchCV(svc, param_grid={'C': C_array})
grid.fit(X_train, y_train)
print('CV error    = ', 1 - grid.best_score_)
print('best C      = ', grid.best_estimator_.C)

#svc = SVC(C=1, kernel='poly', degree=3)
#svc.fit(X_train, y_train)

#err_train = np.mean(y_train != svc.predict(X_train))
#err_cv  = np.mean(y_cv  != svc.predict(X_cv))
#print(err_train, err_cv)
