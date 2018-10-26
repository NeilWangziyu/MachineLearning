import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import pca
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


face_X = []
face_Y = []
rootpath = '/Users/ziyu/资料/我的课程与资料/研一下/machine learning/CroppedYale'
c = 0
for parent, subdir, filenames in os.walk(rootpath):
  i = 0
  print("reading path:",parent)
  for filename in filenames:
    if filename.endswith('.pgm') and not filename.endswith('Ambient.pgm') :
      #image path is each image's path
      image_paths = [os.path.join(rootpath,parent, filename)]
      #print(image_paths[0])
      #count for each person's face image
      i = i + 1
      #print("read image number:", i)
      if image_paths[0]:
        #print(parent)
        #im is the content of image
        im = matplotlib.image.imread(image_paths[0])
        #print("im length：", len(im))
        im = np.array(im)
        scaler = StandardScaler()
        scaler.fit(im)
        im = scaler.transform(im)
        x_train = im.reshape(-1)
        face_X.append(x_train)
        face_Y.append(c)

  print("read total image number:", i)
  if i != 0:
    c = c + 1
  print("read parent number:", c)
print("read parent total number:", c,"and is marked from 0 to",c-1)

print("length of face_X:", len(face_X))
print("features:", len(face_X[0]))
#print(face_Y)

# ##############################################################
# PCA transfer
print('start to PCA')
startPCA = time.clock()
# feature wanted
K=50
# building model，n_components is the number of feature wanted
model = pca.PCA(n_components=K).fit(face_X)
# transform to run PCA
face_X = model.transform(face_X)

finishPCA = (time.clock() - startPCA)
print("PCA Time used:",finishPCA)

X_train, X_test, y_train, y_test = train_test_split(face_X, face_Y, test_size=0.3)
print("length of X_train:", len(X_train))
print("feature used", len(X_train[0]))
#print(y_train)

print('start to train KNN')
startKNN = time.clock()

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
X_result = knn.predict(X_test)
#print(y_test)

finishKNN = (time.clock() - startKNN)
print("KNN Time used:",finishKNN)

#calculate Similiarity
same = 0
for num in range(0,len(y_test)-1):
  if y_test[num] ==  X_result[num]:
    same = same + 1
similiarity = same/len(y_test)
print("similiarity:", same/len(y_test))

#print(confusion_matrix(y_test, X_result))
#print(len(confusion_matrix(y_test, X_result)))
df_confusion = confusion_matrix(y_test, X_result)


# ##############################################################
# plot_confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.viridis):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# ##############################################################
# Compute confusion matrix
df_confusion = confusion_matrix(y_test, X_result)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(df_confusion, classes=range(0,len(df_confusion)),title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(df_confusion, classes=range(0,len(df_confusion)), normalize=True,
#                       title='Feature:'+str(K)+'  Similiarity'+ format(similiarity, '.2%'))

# plot_confusion_matrix(df_confusion, classes=range(0,len(df_confusion)), normalize=True,
#                         title='Feature:ALL'+'  Similiarity'+ format(similiarity, '.2%'))
# plt.show()

# ################################################################
#SVM

print("Fitting the classifier to the training set")
TrainSVMTime = time.clock()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train, y_train)
print("Training done in %0.3fs" % (time.clock() - TrainSVMTime))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

ClassifySVMTime = time.clock()
X_result = clf.predict(X_test)
print("Classify done in %0.3fs" % (time.clock() - ClassifySVMTime))
#calculate Similiarity
same = 0
for num in range(0,len(y_test)-1):
  if y_test[num] ==  X_result[num]:
    same = same + 1
similiarity = same/len(y_test)
print("similiarity:", same/len(y_test))

df_confusion_SVM = confusion_matrix(y_test, X_result)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(df_confusion, classes=range(0,len(df_confusion_SVM)), normalize=True,
                      title='SVM: Feature:'+str(K)+'  Similiarity'+ format(similiarity, '.2%'))
plt.show()



