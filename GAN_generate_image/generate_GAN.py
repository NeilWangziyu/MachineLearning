import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from keras.utils.np_utils import to_categorical
from keras.models import load_model

K.set_image_dim_ordering('th')

# Deterministic output.
# Tired of seeing the same results every time? Remove the line below.


# The results are a little better when the dimensionality of the random vector is only 10.
# The dimensionality has been left at 100 for consistency with other GAN implementations.

def create_digit_sequence(number, image_width=10, min_spacing=0, max_spacing=0):
    """ A function that create an image representing the given number,
        with random spacing between the digits.
        Each digit is randomly sampled from the MNIST dataset.
        Returns an NumPy array representing the image.
        Parameters
        ----------
        number: str
            A string representing the number, e.g. "14543"
        image_width: int
            The image width (in pixel).
        min_spacing: int
            The minimum spacing between digits (in pixel).
        max_spacing: int
            The maximum spacing between digits (in pixel).
    """

    if(len(number) == 0):
        raise ValueError("Please input numbers for transfering")

    number_list = []
    try:
        for each in number:
            number_list.append(int(each))
    except:
        raise ValueError("Please make sure each number is correctly input")

    figsize = (image_width, image_width)

    inputDim = 10

    model_folder = 'model_numb_Gan_mse_sftmx_cc_cc'
    discriminator = load_model('{}/dcgan_discriminator_epoch_75.h5'.format(model_folder))
    generator = load_model('{}/dcgan_generator_epoch_75.h5'.format(model_folder))

    generatedImages = generator.predict(to_categorical(number_list, inputDim))
    # print(generatedImages)
    # dim = (28, 28)

    # plt.imshow(generatedImages[0, 0], interpolation='nearest', cmap='gray_r')
    ## To plot ###
    dim = (1, 10)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generatedImages[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')

    plt.tight_layout()
    # plt.show()
    plt.savefig("digit_sequence.png")
    print("finished")


if __name__ == '__main__':
    create_digit_sequence("1010101", image_width=10)



