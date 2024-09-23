import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

class VAE(tf.keras.Model):
    def __init__(self, latent_dim, batch_size, encoder, decoder):
        """
        Implementation of Variational Autoencoder (VAE) for MNIST.
        Paper (Kingma & Welling): https://arxiv.org/abs/1312.6114.

        :param latent_dim: Dimension of latent space.
        :param batch_size: Number of data points per mini batch.
        :param encoder: function which encodes a batch of inputs to a 
            parameterization of a diagonal Gaussian
        :param decoder: function which decodes a batch of samples from 
            the latent space and returns the corresponding batch of images.
        """
        super(VAE, self).__init__()
        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._encode = encoder
        self._decode = decoder

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        """Create the encoder network."""
        inputs = layers.Input(shape=(28*28,))
        x = layers.Dense(512, activation='relu')(inputs)
        mean = layers.Dense(self._latent_dim)(x)
        logvar = layers.Dense(self._latent_dim)(x)
        model = Model(inputs, [mean, logvar])
        return model

    def _build_decoder(self):
        """Create the decoder network."""
        inputs = layers.Input(shape=(self._latent_dim,))
        x = layers.Dense(512, activation='relu')(inputs)
        outputs = layers.Dense(28*28, activation='sigmoid')(x)
        model = Model(inputs, outputs)
        return model

    def call(self, inputs):
        """Pass the inputs through encoder, sampling, and decoder."""
        mean, logvar = self.encoder(inputs)
        z = self.sample(mean, logvar)
        return self.decoder(z)

    def sample(self, mean, logvar):
        """Reparameterization trick to sample z from latent space."""
        epsilon = tf.random.normal(shape=(self._batch_size, self._latent_dim))
        stddev = tf.exp(0.5 * logvar)
        return mean + stddev * epsilon

    def compute_loss(self, x):
        """Compute the VAE loss."""
        mean, logvar = self.encoder(x)
        z = self.sample(mean, logvar)
        decoded = self.decoder(z)
        # Compute reconstruction loss (Bernoulli)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, decoded), axis=-1)
        )
        # Compute KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)
        )
        return reconstruction_loss + kl_loss

    @tf.function
    def train_step(self, x, optimizer):
        """Single training step."""
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

# Example usage:
latent_dim = 2
batch_size = 64
vae = VAE(latent_dim, batch_size, None, None)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Now, vae.train_step(input_data, optimizer) can be called to perform training steps
