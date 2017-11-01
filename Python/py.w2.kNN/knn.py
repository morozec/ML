>>> import os
>>> os.chdir("D:\OneDrive\py.ex3.kNN")
>>> import pandas
>>> data = pandas.read_csv("wine.data", header = None)
>>> import numpy as np
>>> y = np.array(data[0])
>>> X=[]
>>> for index, row in data.iterrows():
	curr_list=[]
	for j in range(1,14):
		val=row[j]
		curr_list.append(val)
	X.append(curr_list)

>>> X_arr=np.asarray(X)
>>> from sklearn.model_selection import KFold
>>> kFold = KFold(5, True, 42)
>>> from sklearn.neighbors import KNeighborsClassifier
>>> from sklearn.model_selection import cross_val_score

>>> k_max=50
>>> max_av_score = 0
>>> max_av_score_k=0
>>> for i in range(1,k_max+1):
	knc = KNeighborsClassifier(i)
	score = cross_val_score(estimator=knc, X = X_arr, y = y, scoring = 'accuracy', cv = kFold)
	average_score = score.mean()
	if average_score > max_av_score:
		max_av_score = average_score
		max_av_score_k = i

		
		
		
>>> from sklearn.preprocessing import scale
>>> X_arr = scale(X_arr, axis = 0)
>>> max_av_score=0
>>> max_av_score_k=0
>>> for i in range(2,k_max+1):
	knc = KNeighborsClassifier(i)
	score = cross_val_score(estimator=knc, X = X_arr, y = y, scoring = 'accuracy', cv = kFold)
	average_score = score.mean()
	if average_score > max_av_score:
		max_av_score = average_score
		max_av_score_k = i

	

