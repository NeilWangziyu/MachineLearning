from sklearn import datasets, svm
import numpy as np
digits = datasets.load_digits()
X_digits = digits.data
Y_digits = digits.target

svc = svm.SVC(C=1, kernel='linear')
svc.fit(X_digits[:-100], Y_digits[:-100]).score(X_digits[-100:], Y_digits[-100:])
print(svc.score(X_digits[-100:], Y_digits[-100:]))

X_folds = np.array_split(X_digits, 3)
# print(X_folds)
Y_folds = np.array_split(Y_digits, 3)
# print(Y_folds)
scores = list()
for k in range(3):
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)
    Y_train = list(Y_folds)
    Y_test = Y_train.pop(k)
    Y_train = np.concatenate(Y_train)
    scores.append(svc.fit(X_train, Y_train).score(X_test, Y_test))
print(scores)

