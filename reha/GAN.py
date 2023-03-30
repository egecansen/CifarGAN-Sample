import numpy as np
from keras import Sequential
from keras.optimizers import Adam

from reha.Discriminator import build_discriminator
from reha.Generator import build_generator

adam = Adam(learning_rate=0.0002)

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

generator = build_generator()

GAN = Sequential()
discriminator.trainable = False
GAN.add(generator)
GAN.add(discriminator)

GAN.compile(loss='binary_crossentropy', optimizer=adam)

GAN.summary()
