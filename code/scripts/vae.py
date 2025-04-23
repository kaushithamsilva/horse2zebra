import tensorflow as tf
from tensorflow import keras
import numpy as np


@keras.utils.register_keras_serializable()
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


@keras.utils.register_keras_serializable()
class VAE(keras.Model):
    def __init__(self, input_dim, latent_dim, hidden_dim, **kwargs):
        if 'policy' in kwargs and isinstance(kwargs['policy'], str):
            kwargs['policy'] = tf.keras.mixed_precision.Policy(
                kwargs['policy'])

        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # Encoder
        self.encoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=input_dim),

            keras.layers.Conv2D(hidden_dim, 4, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Conv2D(hidden_dim * 2, 4, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Conv2D(hidden_dim * 4, 4, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Conv2D(hidden_dim * 8, 4, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Conv2D(hidden_dim * 8, 4, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Flatten(),
            keras.layers.Dense(hidden_dim * 16, activation='relu'),
            keras.layers.Dense(latent_dim * 2)  # z_mean and z_log_var
        ])

        self.sampling = Sampling()

        # Decoder
        self.decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(latent_dim,)),

            keras.layers.Dense(8 * 8 * hidden_dim * 8, activation='relu'),
            keras.layers.Reshape((8, 8, hidden_dim * 8)),

            keras.layers.Conv2DTranspose(
                hidden_dim * 8, 4, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Conv2DTranspose(
                hidden_dim * 4, 4, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Conv2DTranspose(
                hidden_dim * 2, 4, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Conv2DTranspose(
                hidden_dim, 4, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Conv2DTranspose(
                3, 4, strides=2, padding='same', activation='sigmoid')
        ])

    def encode(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = tf.split(x, num_or_size_splits=2, axis=1)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encode(inputs)
        reconstructed = self.decode(z)
        return reconstructed, z_mean, z_log_var

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_z_mean_embeddings(data, vae_model):
    embeddings = []
    chunk_size = 2000  # Process data in chunks to avoid memory issues
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        transformed_chunk, _, _ = vae_model.encode(chunk)
        embeddings.append(transformed_chunk)

    return np.vstack(embeddings)
