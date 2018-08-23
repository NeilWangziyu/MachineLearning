import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def build_model():
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=(32,32,3)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal'))
    # sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model


if __name__ == '__main__':

    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('size of train data:',len(x_train))
    print('size of test data:',len(x_test))
    # plt.imshow(x_train[0])
    # plt.show()
    # plt.imshow(x_train[1])
    # plt.show()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train.reshape(-1, 32, 32, 3) / 255.
    x_test = x_test.reshape(-1, 32, 32, 3) / 255.

    # build network
    model = build_model()
    print(model.summary())

    # # set callback
    # tb_cb = TensorBoard(log_dir='./lenet', histogram_freq=0)
    # change_lr = LearningRateScheduler(scheduler)
    # cbks = [change_lr,tb_cb]

    # start train
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=100,
              shuffle=True)

    loss, accuracy = model.evaluate(x_test, y_test)

    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)

    # save model
    # print('\nTesting ------------')
    # # Evaluate the model with the metrics we defined earlier
    # loss, accuracy = model.evaluate(x_test, y_test)
    #
    # print('\ntest loss: ', loss)
    # print('\ntest accuracy: ', accuracy)