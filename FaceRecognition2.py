import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np
import time
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten,Dropout
from keras.optimizers import Adam

face_X = []
face_Y = []
rootpath = '/Users/ziyu/PycharmProjects/FaceRecognition3/CroppedYale'
c = 0
startRead = time.clock()
print("start reading data...")
for parent, subdir, filenames in os.walk(rootpath):
  i = 0
  for filename in filenames:
    if filename.endswith('.pgm') and not filename.endswith('Ambient.pgm') :
      image_paths = [os.path.join(rootpath,parent, filename)]
      i = i + 1
      if image_paths[0]:
        #im is the content of image
        im = matplotlib.image.imread(image_paths[0])

        scaler = StandardScaler()
        scaler.fit(im)
        im = scaler.transform(im)
        face_X.append(im)
        face_Y.append(c)

  #print("read total image number:", i)
  if i != 0:
    c = c + 1
  #print("read parent number:", c)
print("read parent total number:", c,"and is marked from 0 to",c-1)
finishREAD = (time.clock() - startRead)
print("READING Time used:",finishREAD)

print("length of face_X:", len(face_X))

X_train, X_test, y_train, y_test = train_test_split(face_X, face_Y, test_size=0.2)
print("length of X_train:", len(X_train))
X_train=np.array(X_train)
X_train = X_train.reshape(-1, 1,192, 168)
X_test=np.array(X_test)
X_test = X_test.reshape(-1, 1,192, 168)
#print(X_train)

y_train = np_utils.to_categorical(y_train,num_classes=c)
#print(y_train)
y_test_origin = y_test
y_test = np_utils.to_categorical(y_test,num_classes=c)
print("number of category:",c)

model = Sequential(
    [
# Conv layer 1 output shape (32, 192, 168)
    Convolution2D(
        batch_input_shape=(None,1,192,168), #1 channel个数
        filters=16,
        kernel_size=5,
        strides=1,
        #normally 1 in convolution layer
        padding='same',  # Padding method
        data_format='Channels_first',          ),
    Activation('relu'),

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
    MaxPooling2D(
        pool_size=2,
        strides=2,
        #normally 2
        padding='same',
        data_format='channels_first',
    ),

# Conv layer 2 output shape (64, 14, 14)
    Convolution2D(
        filters=36,
        kernel_size=5,
        strides=1,
        padding='same',
        data_format='channels_first',
    ),
    Activation('relu'),

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
    MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',
        data_format='channels_first',
    ),

    # Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024) 1024 is an experience number
    Flatten(),
    Dense(512),
    Activation('relu'),

    # Fully connected layer 2 to shape (c) for c classes
    Dense(c),
    Activation('softmax'),

    ]
)

adam = Adam(lr=1e-4)
# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
history = model.fit(X_train, y_train, epochs=2, batch_size=100)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
model.summary()


print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
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


X_result = model.predict(X_test)
#print(X_result)
#print(type(X_result[0]))
X_result_transfer = X_result.argmax(axis=1)
print(X_result_transfer)

df_confusion = confusion_matrix(y_test_origin, X_result_transfer)
plt.figure()
plot_confusion_matrix(df_confusion, classes=range(0,len(df_confusion)), normalize=True,
                      title='confusion matrix for CNN')
plt.show()
