from sklearn.model_selection import KFold
import numpy as np
X = np.array([[1,2], [3,4], [0,0], [1,2], [5,6], [12,9], [9,8]])
Y = np.array([1, 1, 1, 1, 0, 0, 0])
print(X)
print(len(X))
print(X[0])
kf = KFold(n_splits=3, shuffle=False)
print(kf.get_n_splits(X))
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "Test:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]




