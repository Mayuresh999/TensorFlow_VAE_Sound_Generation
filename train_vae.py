from variational_autoencoder import VAE
import os
import numpy as np


LEARNING_RATE = 0.0005
BATCH_SIZE = 256
NUM_EPOCHS = 500

SPECTOGRAMS = "SPECTOGRAMS"

def load_dataset(data_path):
    x_train = []
    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            spectrogram = np.load(file_path)
            x_train.append(spectrogram)
            # print(spectrogram, spectrogram.shape)
    x_train = np.array(x_train[0])
    x_train = x_train[..., np.newaxis]
    # x_train = np.reshape(x_train, (-1, 257, 128, 1))
    print(x_train.shape)
    return x_train


def train(x_train, learning_rate, batch_size, num_epochs):
    autoencoder = VAE(
        input_shape=(257, 128, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2,1)),
        latent_space_dim=128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, num_epochs)
    return autoencoder

if __name__ == '__main__':
    x_train= load_dataset("SPECTOGRAMS")
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS)
    autoencoder.save("model")
    autoencoder2 = autoencoder.load("model")
    autoencoder2.summary()