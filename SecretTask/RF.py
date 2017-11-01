from sklearn import ensemble
import matplotlib.pyplot as plt
rf = ensemble.RandomForestClassifier(n_estimators=5000, random_state=11)
rf.fit(X, y)
#err_train = np.mean(y_train != rf.predict(X_train))
#err_cv  = np.mean(y_cv  != rf.predict(X_cv))

#print(err_train, err_cv)

##feature_names = data.columns
##
##importances = rf.feature_importances_
##indices = np.argsort(importances)[::-1]
##
##d_first = 50
##plt.figure(figsize=(8, 8))
##plt.title("Feature importances")
##plt.bar(range(d_first), importances[indices[:d_first]], align='center')
##plt.xticks(range(d_first), np.array(feature_names)[indices[:d_first]], rotation=90)
##plt.xlim([-1, d_first])
##
##best_features = indices[:9]
##best_features_names = feature_names[best_features]
