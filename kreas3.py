import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import  np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train),(X_test, y_test) = mnist.load_data()
#print(X_train[0])
# data pre-processing 此处，在每一个图像上再套上一个[]
X_train = X_train.reshape(-1, 1,28, 28)/255.
#print(X_train[0])
X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

model = Sequential(
    [
# Conv layer 1 output shape (32, 28, 28)
    Convolution2D(
        batch_input_shape=(None,1,28,28), #1 channel个数
        filters=32,
        kernel_size=5,
        strides=1,
        #normally 1 in convolution layer
        padding='same',  # Padding method
        data_format='Channels_first', #指data的format里面channels 是最后一个
        #换一个申明28,28,1 的格式(颜色在最后的时候) . data_format='channels_last'
    ),
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
        filters=64,
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
    Dense(1024),
    Activation('relu'),

    # Fully connected layer 2 to shape (10) for 10 classes
    Dense(10),
    Activation('softmax'),

    ]
)

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=1, batch_size=64,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
