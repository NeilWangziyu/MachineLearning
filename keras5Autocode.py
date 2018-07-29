import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

# X shape (60,000 28x28), y shape (10,000, )
(x_train, _), (x_test,y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)

# data pre-processing
x_train = x_train.astype('float32') / 255. - 0.5
x_test = x_test.astype('float32') / 255. - 0.5
print('x_train.shape[0] is', x_train.shape[0])
print('x_test.shape[0] is', x_test.shape[0])

x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0],-1))

print(x_train.shape)
print(x_test.shape)

# in order to plot in a 2D figure
encoding_dim = 2

#this is our input placeholder
input_img = Input(shape=(784, ))

#encoder layer
encoded = Dense(128,activation='relu')(input_img)
encoded = Dense(64,activation='relu')(encoded)
encoded = Dense(10,activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

#decoder layer
decoded = Dense(10,activation='relu')(encoder_output)
decoded = Dense(64,activation='relu')(decoded)
decoded = Dense(128,activation='relu')(decoded)
decoded = Dense(784,activation='tanh')(decoded)

#construct the autoencoder model
autoencoder = Model(input=input_img, output=decoded)

#construct the encoder model for ploting
encoder = Model(input=input_img, output=encoder_output)

#compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

#training
autoencoder.fit(
    x_train,x_train,
    epochs=20,
    batch_size=256,
    shuffle=True
)

#plotting
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:,0], encoded_imgs[:,1],c=y_test)
plt.colorbar()
plt.show()