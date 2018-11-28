from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

iris = load_iris()
X_train = iris.data
y_train = iris.target
X_test = iris.data[50:60]
y_test = iris.target[50:60]
X_train_less = iris.data[:, 1:3]
print(X_train_less)
# print(X_train)
# print(y_train)
clf = ExtraTreesClassifier(n_estimators=400, n_jobs=8, verbose=10)
clf.fit(X_train, y_train)
y_pre = clf.predict(X_test)
print(y_pre)
print(y_test)

clf.fit(X_train_less, y_train)
y_pre = clf.predict(X_test)