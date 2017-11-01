>>> from sklearn import datasets
>>> newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> vectorizer = TfidfVectorizer()
>>> X = vectorizer.fit_transform(newsgroups.data)
>>> feature_mapping = vectorizer.get_feature_names()

>>> from sklearn.grid_search import GridSearchCV

>>> from sklearn.model_selection import GridSearchCV
>>> from sklearn.model_selection import KFold
>>> import numpy as np
>>> grid = {'C': np.power(10.0, np.arange(-5, 6))}
>>> cv = KFold(5, True, random_state=241)

>>> from sklearn.svm import SVC
>>> clf = SVC(C =1, kernel='linear', random_state=241)
>>> gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
>>> y = newsgroups.target
>>> gs.fit(X, y)
GridSearchCV(cv=KFold(n_splits=5, random_state=241, shuffle=True),
       error_score='raise',
       estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=241, shrinking=True,
  tol=0.001, verbose=False),
       fit_params={}, iid=True, n_jobs=1,
       param_grid={'C': array([  1.00000e-05,   1.00000e-04,   1.00000e-03,   1.00000e-02,
         1.00000e-01,   1.00000e+00,   1.00000e+01,   1.00000e+02,
         1.00000e+03,   1.00000e+04,   1.00000e+05])},
       pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
       scoring='accuracy', verbose=0)

>>> clf = SVC(C =1, kernel='linear', random_state=241)
>>> clf.fit(X,y)
SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=241, shrinking=True,
  tol=0.001, verbose=False)
>>> coeffs = clf.coef_

>>> result=[]

>>> for i in range(0,10):
	max_value=0
	max_index = 0	
	for j in range(0, coeffs.shape[1]):
		coeff = abs(coeffs[0,j])
		if (coeff > max_value) and result.count(j)==0:
			max_value = coeff
			max_index = j
	result.append(max_index)
	print(result)

>>> result_str = []
>>> for i in result:
	result_str.append(feature_mapping[i])
	
>>> result_str.sort()
 
