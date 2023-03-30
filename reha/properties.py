import os

try:
    os.mkdir("../generated_images")
except FileExistsError:
    pass

img_width = 32
img_height = 32
channels = 3
img_shape = (img_width, img_height, channels)
latent_dim = 100

