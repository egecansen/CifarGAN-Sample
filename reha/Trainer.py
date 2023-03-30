import numpy as np
from keras.datasets import cifar10, mnist

from reha.properties import latent_dim
from reha.GAN import GAN, generator, discriminator
from utils.SaveImgs import save_imgs


def train(epochs, batch_size=64, save_interval=200):
    (X_train, _), (_, _) = cifar10.load_data()

    # Rescaling the data
    X_train = X_train / 127.5 - 1.

    bat_per_epo = int(X_train.shape[0] / batch_size)

    # Create Y label for NN
    valid = np.ones((batch_size, 1))
    fakes = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        for j in range(bat_per_epo):
            # Get Random Batch
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # print("Shape: {0}".format(imgs.shape))

            # Generate Fake Images
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            gen_imgs = generator.predict(noise)

            # Train discriminator
            d_loss_real = discriminator.train_on_batch(imgs, valid)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fakes)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, latent_dim))

            # inverse y label
            g_loss = GAN.train_on_batch(noise, valid)

            print("******* %d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            if (j % save_interval) == 0:
                save_imgs(epoch)
