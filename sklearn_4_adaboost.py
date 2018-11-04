from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

iris = load_iris()
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print(scores.mean())

y_train_pred = cross_val_predict(clf, iris.data, iris.target, cv=5)
conf_mx = confusion_matrix(iris.target, y_train_pred)
print(conf_mx)

plt.matshow(conf_mx)
plt.show()