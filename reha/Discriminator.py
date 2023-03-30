from keras import Sequential
from keras.layers import LeakyReLU, Conv2D, Flatten, Dropout, Dense

from reha.properties import img_shape


def build_discriminator():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=img_shape))

    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3, 3), padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, (3, 3), padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, (3, 3), padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, (3, 3), padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, (3, 3), padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.4))  # Drops 40% of neurons?
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    return model