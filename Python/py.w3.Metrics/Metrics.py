Python 3.6.0 |Anaconda 4.3.1 (64-bit)| (default, Dec 23 2016, 11:57:41) [MSC v.1900 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import os
>>> os.chdir("D:\OneDrive\ML\py.w3.Metrics")
>>> import pandas
import numpy 
>>> import numpy as np
>>> data = pandas.read_csv("classification.csv")
>>> trues = np.array(data['true'])
>>> preds = np.array(data['pred'])
>>> l = len(preds)
>>> tp = 0
>>> fp = 0
>>> fn = 0
>>> tn = 0
>>> for i in range(l):
	if trues[i] == 1 and preds[i] == 1:
		tp = tp + 1
	if trues[i] == 0 and preds[i] == 0:
		tn = tn + 1
	if trues[i] == 1 and preds[i] == 0:
		fn = fn + 1
	if trues[i] == 0 and preds[i] == 1:
		fp = fp + 1

>>> import sklearn.metrics
>>> accuracy_score = sklearn.metrics.accuracy_score(trues, preds)
>>> precision_score = sklearn.metrics.precision_score(trues, preds)
>>> recall_score = sklearn.metrics.recall_score(trues, preds)
>>> f1_score = sklearn.metrics.f1_score(trues, preds)


>>> scores_data = pandas.read_csv("scores.csv")
>>> score_logreg = np.array(scores_data['score_logreg'])
>>> score_svm = np.array(scores_data['score_svm'])
>>> score_knn = np.array(scores_data['score_knn'])
>>> score_tree = np.array(scores_data['score_tree'])
>>> score_trues = np.array(scores_data['true'])
>>> roc_auc_score_logreg = sklearn.metrics.roc_auc_score(score_trues,score_logreg)
>>> roc_auc_score_svm = sklearn.metrics.roc_auc_score(score_trues,score_svm)
>>> roc_auc_score_knn = sklearn.metrics.roc_auc_score(score_trues,score_knn)
>>> roc_auc_score_tree = sklearn.metrics.roc_auc_score(score_trues,score_tree)


>>> def GetMaxPrecision(precision, recall, thresholds):
	l = len(recall)
	max_precision = 0
	for i in range(l):
		if recall[i] >= 0.7:
			if precision[i] > max_precision:
				max_precision = precision[i]
	return max_precision

>>> precision_recall_curve_logreg = sklearn.metrics.precision_recall_curve(score_trues,score_logreg)
>>> maxP = GetMaxPrecision(precision_recall_curve_logreg[0], precision_recall_curve_logreg[1], precision_recall_curve_logreg[2])
>>> precision_recall_curve_svm = sklearn.metrics.precision_recall_curve(score_trues,score_svm)
>>> maxP_svm = GetMaxPrecision(precision_recall_curve_svm[0], precision_recall_curve_svm[1], precision_recall_curve_svm[2])
>>> precision_recall_curve_knn = sklearn.metrics.precision_recall_curve(score_trues,score_knn)
>>> maxP_knn = GetMaxPrecision(precision_recall_curve_knn[0], precision_recall_curve_knn[1], precision_recall_curve_knn[2])
>>> precision_recall_curve_tree = sklearn.metrics.precision_recall_curve(score_trues,score_tree)
>>> maxP_tree = GetMaxPrecision(precision_recall_curve_tree[0], precision_recall_curve_tree[1], precision_recall_curve_tree[2])
