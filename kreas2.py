import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_tain 是重数组，y_train是数组从0-9
# print(X_train[1])
# print(len(X_train))
# print (y_train)
# print(len(y_train))
print(X_train.shape)
# data pre-processing
X_train = X_train.reshape(X_train.shape[0],-1) / 255.
#X_train 全部变为一行，60000 * 784 数组
# -1 表示自动推测出，X_train.reshape(60000,-1)
# print(X_train[0])
# print(len(X_train[0]))
X_test= X_test.reshape(X_test.shape[0],-1) / 255.
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)
print(X_train.shape)

# Another way to build your neural net
model = Sequential(
    [
        Dense(64,input_dim=784),
        Activation('relu'),
        Dense(10),
        Activation('softmax'),

    ]
)

# Another way to define your optimizer
remsprop = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)

# We add metrics to get more results you want to see
model.compile(
    optimizer=remsprop,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


print('Training ------------')
# Another way to train the model
model.fit(X_train,y_train,epochs=3,batch_size=64)
#batch_size epochs can be modified

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss,accuracy = model.evaluate(X_test,y_test)

print('loss:', loss)
print('test accuracy:', accuracy)


