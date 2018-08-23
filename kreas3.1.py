import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import  np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten,Dropout
from keras.optimizers import Adam

# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train),(X_test, y_test) = mnist.load_data()
#print(X_train[0])
# data pre-processing 此处，在每一个图像上再套上一个[]
#print(X_train)
X_train = X_train.reshape(-1, 1, 28, 28)/255.
#print(X_train[0])
X_test = X_test.reshape(-1, 1, 28, 28)/255.
print(y_train)
y_train = np_utils.to_categorical(y_train,num_classes=10)
print(y_train)

y_test = np_utils.to_categorical(y_test,num_classes=10)

model = Sequential()
model.add(Convolution2D(30,(3,3),input_shape=(1,28,28),activation='relu',data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(15,(3,3),activation='relu',data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add((Dropout(0.2)))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=200)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
