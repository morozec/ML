>>> import os
>>> os.chdir("D:\OneDrive\ML\py.w4.LR.Text")
>>> import pandas
>>> data_train = pandas.read_csv("salary-train.csv")
>>> data_test = pandas.read_csv("salary-test-mini.csv")
>>> import numpy as np

>>> data_train['FullDescription'] = data_train['FullDescription'].str.lower()
>>> data_train['LocationNormalized'] = data_train['LocationNormalized'].str.lower()
>>> data_train['ContractTime'] = data_train['ContractTime'].str.lower()

>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> vectorizer = TfidfVectorizer(min_df=5)
>>> fd_vectorized = vectorizer.fit_transform(data_train['FullDescription'])

>>> from sklearn.feature_extraction import DictVectorizer
>>> data_train['LocationNormalized'].fillna('nan', inplace=True)
>>> data_train['ContractTime'].fillna('nan', inplace=True)
>>> enc = DictVectorizer()
>>> X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))

>>> from scipy.sparse import hstack
>>> data_train_result = hstack([fd_vectorized,X_train_categ])

>>> y_train = np.array(data_train['SalaryNormalized'])

>>> from sklearn.linear_model import Ridge
>>> ridge = Ridge(alpha=1, random_state=241)
>>> ridge.fit(data_train_result, y_train)
Ridge(alpha=1, copy_X=True, fit_intercept=True, max_iter=None,
   normalize=False, random_state=241, solver='auto', tol=0.001)

   
>>> data_test['FullDescription'] = data_test['FullDescription'].str.lower()
>>> data_test['ContractTime'] = data_test['ContractTime'].str.lower()
>>> data_test['LocationNormalized'] = data_test['LocationNormalized'].str.lower()

>>> data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

>>> data_test['LocationNormalized'].fillna('nan', inplace=True)
>>> data_test['ContractTime'].fillna('nan', inplace=True)

>>> fd_vectorized_test = vectorizer.transform(data_test['FullDescription'])

>>> X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

>>> data_test_result = hstack([fd_vectorized_test,X_test_categ])
>>> y_test = ridge.predict(data_test_result)

