#---Create Algoritm Cancer diagnosis in Machine Learning---

#Importing Librarys Machine Learning AI-
import sklearn
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

#Starting Programming Machine Learning-
bcd = datasets.load_breast_cancer()

x = bcd.data
y = bcd.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train, y_train)
y_prediction = knn.predict(x_test)

print(confusion_matrix(y_test, y_prediction, [0, 1]))
print(classification_report(y_test, y_prediction))

log = LogisticRegression()
log.fit(x_train, y_train)
y_pred = log.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm

from sklearn.preprocessing import normalize
cm = normalize(cm,norm='l1',axis=1)
cm_df = pd.DataFrame(cm, columns=bcd.target_names, index=bcd.target_names)
print(cm_df)

from sklearn.metrics import roc_curve
y_pred_prob = log.predict_proba(x_test)[:,1]
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr,tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_prob)

from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors':np.arange(1,50)} # میدھیم قرار دیکشنری داخل را ھا ھایپرپارامتر
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv = 5)
knn_cv.fit(x,y)
print(knn_cv.best_params_)
print(knn_cv.best_score_) # Returns the mean accuracy on the given test data and label

from scipy.stats import randint # randint(1, 9).rvs(2)
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
#GridSearchCV can be computationally expensive, especially if you are searching over a
large hyperparameter space and dealing with multiple hyperparameters
param = {"max_depth": [3, None],
 "max_features": randint(1, 9), # [2, 4, 6, 7]
 "min_samples_leaf": randint(1, 9)}
#Dictionary with parameters names (string) as keys and distributions or lists of parame
ters to try.
#Distributions must provide a rvs method for sampling
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree, param, cv=5) #CV=None, to use the default 3-fold cro
ss validation,
tree_cv.fit(x_train, y_train)
print(tree_cv.best_params_)
print(tree_cv.best_score_)
y_pred = tree_cv.predict(x_test)
score = tree_cv.score(x_test, y_test)
print(score)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)

