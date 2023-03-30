from keras import Sequential
from keras.layers import Dense, LeakyReLU, Reshape, Conv2DTranspose, Conv2D

from reha.properties import latent_dim


def build_generator():
    model = Sequential()

    model.add(Dense(256 * 4 * 4, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Reshape((4, 4, 256)))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))

    model.summary()

    return model
