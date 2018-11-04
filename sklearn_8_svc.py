from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

(X, y) = make_moons(200, noise=0.2)

poly_kernel_Svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
))

print(poly_kernel_Svm_clf.fit(X, y))

import numpy as np
import matplotlib.pyplot as plt

xx, yy = np.meshgrid(np.arange(-2,3,0.01), np.arange(-1,2,0.01))
y_new=poly_kernel_Svm_clf.predict(np.c_[xx.ravel(),yy.ravel()])
plt.contourf(xx, yy, y_new.reshape(xx.shape),cmap="PuBu")
# 等高线图：contourf
plt.scatter(X[:,0],X[:,1],marker="o",c=y)
plt.show()



rbf_kernel_svm_clf = Pipeline((
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
))
rbf_kernel_svm_clf.fit(X, y)
#画出决策线

xx, yy = np.meshgrid(np.arange(-2,3,0.01), np.arange(-1,2,0.01))
y_new=rbf_kernel_svm_clf.predict(np.c_[xx.ravel(),yy.ravel()])
plt.contourf(xx, yy, y_new.reshape(xx.shape),cmap="PuBu")
plt.scatter(X[:,0],X[:,1],marker="o",c=y)
plt.show()