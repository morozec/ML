>>> import os
>>> os.chdir("D:\OneDrive\ML\py.w4.PCA")
>>> import pandas
>>> data=pandas.read_csv("close_prices.csv")

>>> from sklearn.decomposition import PCA
>>> pca=PCA(n_components=10)

>>> X=data[np.arange(1, 31)]

>>> X_t = pca.transform(X)
>>> X.shape
(374, 30)
>>> X_t.shape
(374, 10)
>>> dj_data = pandas.read_csv("djia_index.csv")

>>> from numpy import corrcoef

>>> X_t_0 = X_t[:,0]
>>> dj = np.array(dj_data['^DJI'])
>>> cc = corrcoef(X_t_0,dj)

>>> abs_0 = abs(pca.components_[0])
>>> max_index = np.argmax(abs_0)
>>> data[[27]]

