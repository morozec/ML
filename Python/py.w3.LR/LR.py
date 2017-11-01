Python 3.6.0 |Anaconda 4.3.1 (64-bit)| (default, Dec 23 2016, 11:57:41) [MSC v.1900 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import os
>>> os.chdir("D:\OneDrive\ML\py.w3.LR")
>>> import pandas
>>> data = pandas.read_csv("data-logistic.csv", header = None)
>>> y = np.array(data[0])
>>> x1 = np.array(data[1])
>>> x2 = np.array(data[2])
>>> l = len(y)
>>> import math
>>> def dist(x1, y1, x2, y2):
	return math.sqrt(pow(x1-x2,2) + pow(y1-y2,2))

>>> from sklearn.metrics import roc_auc_score

>>> def GetW1(w1,w2,y,x1,x2,C):
	l = len(y)
	summa = 0;
	for i in range(l):
		summa = summa + y[i]*x1[i]*(1 - 1 /(1 + math.exp(-y[i]*(w1*x1[i]+w2*x2[i]))))
	new_w1 = w1 + k/l*summa - k*C*w1
	return new_w1
	
>>> def GetW2(w1,w2,y,x1,x2,C):
	l = len(y)
	summa = 0;
	for i in range(l):
		summa = summa + y[i]*x2[i]*(1 - 1 /(1 + math.exp(-y[i]*(w1*x1[i]+w2*x2[i]))))
	new_w2 = w2 + k/l*summa - k*C*w2
	return new_w2

>>> k=0.1
>>> iter_index = 0
>>> w1=0
>>> w2=0
>>> C=0
>>> while iter_index < 10000:
	new_w1 = GetW1(w1, w2, y, x1, x2, C)
	new_w2 = GetW2(w1,w2,y,x1,x2,C)
	d = dist(w1,w2,new_w1,new_w2)
	if d < eps:
		break
	w1 = new_w1
	w2 = new_w2
	iter_index = iter_index + 1

>>> a = []
>>> for i in range(l):
	a_tmp = 1 / (1 + math.exp(-w1*x1[i]-w2*x2[i]))
	a.append(a_tmp)
>>> score = roc_auc_score(y, a)

>>> iter_index = 0
>>> w1=0
>>> w2=0
>>> C=10
>>> while iter_index < 10000:
	new_w1 = GetW1(w1, w2, y, x1, x2, C)
	new_w2 = GetW2(w1,w2,y,x1,x2,C)
	d = dist(w1,w2,new_w1,new_w2)
	if d < eps:
		break
	w1 = new_w1
	w2 = new_w2
	iter_index = iter_index + 1

>>> a_C=[]
>>> for i in range(l):
	a_tmp = 1 / (1 + math.exp(-w1*x1[i]-w2*x2[i]))
	a_C.append(a_tmp)
	
>>> score_C = roc_auc_score(y, a_C)