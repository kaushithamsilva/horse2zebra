import tensorflow as tf
import numpy as np


class Hyperplane:
    def __init__(self, model):
        self.model = model
        # Get the weights and biases from the model's dense layer
        W = self.model.layers[0].kernel
        b = self.model.layers[0].bias   # Shape: (2,)

        # Calculate the hyperplane parameters.  Note the transpose and slicing.
        w = W[:, 0] - W[:, 1]   # Normal vector: (latent_dim,)
        b = b[0] - b[1]       # Offset: scalar

        # Normalize w
        w = w / tf.norm(w)

        self.w = w
        self.b = b

    def get_hyplerplane_params(self):
        return self.w, self.b

    def get_mirror_image(self, z):
        # Calculate the projection of x onto the hyperplane
        z_dist = np.dot(z, self.w) + self.b
        z_mirror = z - 2 * z_dist * self.w
        return z_mirror


def get_hyperplane(domain_discriminator):
    # Get the weights and biases from the domain_discriminator's dense layer
    # Shape: (latent_dim, 2) in TensorFlow
    W = domain_discriminator.layers[0].kernel
    b = domain_discriminator.layers[0].bias   # Shape: (2,)

    # Calculate the hyperplane parameters.  Note the transpose and slicing.
    w = W[:, 0] - W[:, 1]   # Normal vector: (latent_dim,)
    b = b[0] - b[1]       # Offset: scalar

    # Normalize w (optional, but often helpful)
    w = w / tf.norm(w)
    return w, b
