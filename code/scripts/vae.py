import tensorflow as tf
from tensorflow import keras
import numpy as np

# Keep the Sampling layer as defined previously


@keras.utils.register_keras_serializable()
class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # Use tf.random.normal for generating random noise
        epsilon = tf.random.normal(shape=(batch, dim))
        # Calculate z using the reparameterization trick
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        # Implement get_config to enable serialization
        config = super(Sampling, self).get_config()
        return config


@keras.utils.register_keras_serializable()
class VAE(keras.Model):
    """
    Convolutional Variational Autoencoder (VAE) for 256x256x3 images.

    Args:
        input_dim (tuple): Dimensions of the input image (e.g., (256, 256, 3)).
        latent_dim (int): Dimensionality of the latent space.
        base_filters (int): Number of filters in the first convolutional layer.
                            Subsequent layers will increase/decrease filters based on this.
        **kwargs: Additional keyword arguments for keras.Model.
    """

    def __init__(self, input_dim=(256, 256, 3), latent_dim=256, base_filters=64, **kwargs):
        # Handle policy before super() call if needed (e.g., for mixed precision)
        if 'policy' in kwargs and isinstance(kwargs['policy'], str):
            kwargs['policy'] = tf.keras.mixed_precision.Policy(
                kwargs['policy'])

        super(VAE, self).__init__(**kwargs)

        # --- Input Validation ---
        if not (isinstance(input_dim, tuple) and len(input_dim) == 3):
            raise ValueError(
                "`input_dim` must be a tuple of length 3 (height, width, channels).")
        if not (isinstance(latent_dim, int) and latent_dim > 0):
            raise ValueError("`latent_dim` must be a positive integer.")
        if not (isinstance(base_filters, int) and base_filters > 0):
            raise ValueError("`base_filters` must be a positive integer.")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.base_filters = base_filters
        self.final_conv_shape = None  # Will be determined by encoder output

        # === Encoder ===
        # Input: (256, 256, 3)
        encoder_inputs = keras.layers.Input(shape=self.input_dim)
        x = encoder_inputs

        # Block 1: 256 -> 128
        x = keras.layers.Conv2D(self.base_filters, (3, 3),
                                strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        # tf.debugging.check_numerics(x, "Encoder Block 1 Output") # Optional Debugging

        # Block 2: 128 -> 64
        x = keras.layers.Conv2D(
            self.base_filters * 2, (3, 3), strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        # tf.debugging.check_numerics(x, "Encoder Block 2 Output")

        # Block 3: 64 -> 32
        x = keras.layers.Conv2D(
            self.base_filters * 4, (3, 3), strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        # tf.debugging.check_numerics(x, "Encoder Block 3 Output")

        # Block 4: 32 -> 16
        x = keras.layers.Conv2D(
            self.base_filters * 8, (3, 3), strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        # tf.debugging.check_numerics(x, "Encoder Block 4 Output")

        # Block 5: 16 -> 8
        # Keep filters same or increase slightly
        x = keras.layers.Conv2D(
            self.base_filters * 8, (3, 3), strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        # tf.debugging.check_numerics(x, "Encoder Block 5 Output")

        # --- Flatten and Dense Layers ---
        # Shape before flatten will be (batch_size, 8, 8, base_filters * 8)
        # Store shape for decoder: (8, 8, base_filters * 8)
        self.final_conv_shape = tf.shape(x)[1:]
        x = keras.layers.Flatten()(x)
        # tf.debugging.check_numerics(x, "Encoder Flatten Output")

        # Optional Dense layer before latent space
        # x = keras.layers.Dense(self.base_filters * 16, activation='relu')(x) # Example intermediate dense
        # tf.debugging.check_numerics(x, "Encoder Dense Output")

        # --- Latent Space ---
        # Output two vectors: z_mean and z_log_var
        z_mean = keras.layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = keras.layers.Dense(self.latent_dim, name="z_log_var")(x)
        # tf.debugging.check_numerics(z_mean, "z_mean Output")
        # tf.debugging.check_numerics(z_log_var, "z_log_var Output")

        # Use Sampling layer to get z
        z = Sampling()([z_mean, z_log_var])
        # tf.debugging.check_numerics(z, "Sampled z Output")

        # Instantiate the encoder model
        self.encoder = keras.Model(
            encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        # Determine the shape dynamically after building the encoder (needed for decoder)
        # We need the shape *before* flatten for the decoder's first Dense layer
        # Example: If input is (None, 256, 256, 3), output of last conv is (None, 8, 8, base_filters*8)
        self._determine_final_conv_shape()

        # === Decoder ===
        # Input: latent vector z (shape: (batch_size, latent_dim))
        decoder_inputs = keras.layers.Input(shape=(self.latent_dim,))
        # Calculate the number of units needed to reshape back to the final encoder conv shape
        # Example: 8 * 8 * (base_filters * 8)
        dense_units = np.prod(self.final_conv_shape)
        x = keras.layers.Dense(dense_units, use_bias=False)(decoder_inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        # tf.debugging.check_numerics(x, "Decoder Dense Output")

        # Reshape to match the shape before Flatten in the encoder
        x = keras.layers.Reshape(self.final_conv_shape)(x)
        # tf.debugging.check_numerics(x, "Decoder Reshape Output")

        # --- Transposed Convolution Blocks ---
        # Block 1: 8 -> 16
        x = keras.layers.Conv2DTranspose(
            self.base_filters * 8, (3, 3), strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        # tf.debugging.check_numerics(x, "Decoder Block 1 Output")

        # Block 2: 16 -> 32
        x = keras.layers.Conv2DTranspose(
            self.base_filters * 4, (3, 3), strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        # tf.debugging.check_numerics(x, "Decoder Block 2 Output")

        # Block 3: 32 -> 64
        x = keras.layers.Conv2DTranspose(
            self.base_filters * 2, (3, 3), strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        # tf.debugging.check_numerics(x, "Decoder Block 3 Output")

        # Block 4: 64 -> 128
        x = keras.layers.Conv2DTranspose(
            self.base_filters, (3, 3), strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        # tf.debugging.check_numerics(x, "Decoder Block 4 Output")

        # Block 5: 128 -> 256
        # Final layer to reconstruct the image
        # Use 'sigmoid' activation assuming input images are normalized [0, 1]
        # Output channels should match input (3 for RGB)
        decoder_outputs = keras.layers.Conv2DTranspose(
            self.input_dim[-1], (3, 3), strides=2, padding='same', activation='sigmoid')(x)
        # tf.debugging.check_numerics(decoder_outputs, "Decoder Final Output")

        # Instantiate the decoder model
        self.decoder = keras.Model(
            decoder_inputs, decoder_outputs, name="decoder")

    def _determine_final_conv_shape(self):
        """ Helper function to find the shape before flatten in the encoder """
        # Find the Flatten layer
        flatten_layer = None
        for layer in self.encoder.layers:
            if isinstance(layer, keras.layers.Flatten):
                flatten_layer = layer
                break
        if flatten_layer is None:
            raise RuntimeError("Could not find Flatten layer in the encoder.")
        # The shape we need is the output shape of the layer *before* Flatten
        # Keras Functional API layers store input/output tensors.
        # We trace back from the flatten layer's input tensor to find its shape.
        # Get the symbolic tensor feeding into Flatten
        pre_flatten_tensor = flatten_layer.input
        # Get shape excluding batch dim
        self.final_conv_shape = tuple(pre_flatten_tensor.shape[1:])
        print(f"Determined final encoder conv shape: {self.final_conv_shape}")

    def call(self, inputs):
        """ Forward pass of the VAE """
        # Check input data for NaNs/Infs (optional but good practice)
        # tf.debugging.check_numerics(inputs, "VAE Input")

        # Encode the input to get latent variable parameters and sampled z
        z_mean, z_log_var, z = self.encoder(inputs)

        # Decode the latent vector z to reconstruct the input
        reconstructed = self.decoder(z)

        # Return reconstructed image and latent variable parameters
        return reconstructed, z_mean, z_log_var

    # Expose encode/decode methods for convenience
    def encode(self, x):
        """ Encodes input data x and returns latent parameters and sampled z. """
        z_mean, z_log_var, z = self.encoder(x)
        return z_mean, z_log_var, z

    def decode(self, z):
        """ Decodes latent vector z into reconstructed data. """
        return self.decoder(z)

    def get_config(self):
        """ Serialization config """
        config = super(VAE, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "base_filters": self.base_filters,
            # Note: final_conv_shape is determined dynamically, no need to save explicitly
        })
        return config

    @classmethod
    def from_config(cls, config):
        """ Deserialization method """
        # Pop custom args before passing to constructor
        input_dim = config.pop("input_dim", (256, 256, 3))
        latent_dim = config.pop("latent_dim", 256)
        base_filters = config.pop("base_filters", 64)
        # Pass remaining config to the parent class or handle appropriately
        return cls(input_dim=input_dim, latent_dim=latent_dim, base_filters=base_filters, **config)


# --- Example Usage ---
if __name__ == '__main__':
    img_height, img_width, img_channels = 256, 256, 3
    latent_dimension = 256
    base_filter_count = 64

    # Create the VAE model
    vae = VAE(input_dim=(img_height, img_width, img_channels),
              latent_dim=latent_dimension,
              base_filters=base_filter_count)

    # Print model summaries
    print("\nEncoder Summary:")
    vae.encoder.summary()
    print("\nDecoder Summary:")
    vae.decoder.summary()
    print("\nFull VAE Summary (via build):")
    # Build the model with a sample input shape to see the full summary
    vae.build(input_shape=(None, img_height, img_width, img_channels))
    vae.summary()

    # --- Test with dummy data ---
    print("\nTesting with dummy data...")
    batch_size = 4
    dummy_images = tf.random.normal(
        (batch_size, img_height, img_width, img_channels))
    # Ensure input data is in the range [0, 1] if using sigmoid output
    dummy_images = (dummy_images - tf.reduce_min(dummy_images)) / \
        (tf.reduce_max(dummy_images) - tf.reduce_min(dummy_images))

    # Forward pass
    reconstructed, z_mean, z_log_var = vae(dummy_images)
    print(f"Input shape: {dummy_images.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Z_mean shape: {z_mean.shape}")
    print(f"Z_log_var shape: {z_log_var.shape}")

    # Test encode/decode separately
    z_mean_enc, z_log_var_enc, z_enc = vae.encode(dummy_images)
    decoded_z = vae.decode(z_enc)
    print(f"Encoded z shape: {z_enc.shape}")
    print(f"Decoded z shape: {decoded_z.shape}")

    # Check if reconstruction shapes match
    assert reconstructed.shape == dummy_images.shape, "Reconstruction shape mismatch!"
    assert decoded_z.shape == dummy_images.shape, "Separate decode shape mismatch!"
    print("Dummy data test passed.")

    # --- Test Serialization ---
    print("\nTesting serialization...")
    try:
        config = vae.get_config()
        new_vae = VAE.from_config(config)
        # Re-build the new model
        new_vae.build(input_shape=(None, img_height, img_width, img_channels))
        # Test forward pass on new model
        reconstructed_new, _, _ = new_vae(dummy_images)
        assert reconstructed_new.shape == dummy_images.shape, "Serialization shape mismatch!"
        print("Serialization test passed.")
        # Optionally print new model summary
        # print("\nReloaded VAE Summary:")
        # new_vae.summary()
    except Exception as e:
        print(f"Serialization test failed: {e}")
