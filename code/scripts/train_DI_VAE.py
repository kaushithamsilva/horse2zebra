import tensorflow as tf
import numpy as np
import model_utils

# Assuming this function is defined elsewhere and returns normalized w, b
from hyperplane import get_hyperplane


"""
Use the VAE models and train a Domain Informed - VAE (DI-VAE)
"""
SAVE_PATH = '../../models/di_vae/'
CHECKPOINT_PATH = SAVE_PATH+'checkpoints/'
EPOCH_CHECKPOINT = 50


# Load previous epoch for resuming training, set to 0 if starting fresh
PREVIOUS_EPOCH = 5000


def classification_loss(labels, predictions):
    # Add numerical stability check for predictions
    tf.debugging.check_numerics(
        predictions, "Predictions have NaN/Inf in classification_loss")
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, predictions)


def get_mean_matching_loss(z_real, z_mirror, domain_labels):
    """
    z_real:       (B, D) original latents
    z_mirror:     (B, D) reflected latents
    domain_labels:(B,) values 0=horse,1=zebra
    """
    # Masks
    is_horse = tf.equal(domain_labels, 0)
    is_zebra = tf.equal(domain_labels, 1)

    # Real sets
    z_horse = tf.boolean_mask(z_real, is_horse)   # real horses
    z_zebra = tf.boolean_mask(z_real, is_zebra)   # real zebras

    # Mirror sets
    zm_horse = tf.boolean_mask(z_mirror, is_horse)  # horse→mirror
    zm_zebra = tf.boolean_mask(z_mirror, is_zebra)  # zebra→mirror

    loss = 0.0
    count = 0

    # Horse→Mirror should match real zebras
    if tf.shape(zm_horse)[0] > 0 and tf.shape(z_zebra)[0] > 0:
        mean_real_zebra = tf.reduce_mean(z_zebra, axis=0)
        mean_mirror_horse = tf.reduce_mean(zm_horse, axis=0)
        loss += tf.reduce_mean(tf.square(mean_real_zebra - mean_mirror_horse))
        count += 1

    # Zebra→Mirror should match real horses
    if tf.shape(zm_zebra)[0] > 0 and tf.shape(z_horse)[0] > 0:
        mean_real_horse = tf.reduce_mean(z_horse, axis=0)
        mean_mirror_zebra = tf.reduce_mean(zm_zebra, axis=0)
        loss += tf.reduce_mean(tf.square(mean_real_horse - mean_mirror_zebra))
        count += 1

    if count > 0:
        loss /= tf.cast(count, tf.float32)

    return loss


def cycle_weight(epoch):
    """
    Cycle weight function. The weight is 0 for the first few epochs and then increases linearly to max_limit.
    """
    return 0.01  # deactivate cycle loss for now


def kl_weight(epoch):
    return 0.0001  # constant kl loss


@tf.function
def train_step_di(vae_model, domain_discriminator, x, d, optimizer, epoch, clip_norm=1.0):
    """
    Train step for DI-VAE with numerical checks and gradient clipping.
    Note: Renamed from train_step_ci_di to train_step_di for clarity.
    """
    # Check input data first
    tf.debugging.check_numerics(x, "Input x has NaN/Inf")

    with tf.GradientTape() as tape:
        # --- VAE Forward Pass ---
        # Error originates here, indicating issue within vae_model (likely decoder)
        reconstructed, z_mean, z_log_var = vae_model(x)
        tf.debugging.check_numerics(reconstructed, "Reconstructed has NaN/Inf")
        tf.debugging.check_numerics(z_mean, "z_mean has NaN/Inf")
        tf.debugging.check_numerics(z_log_var, "z_log_var has NaN/Inf")

        reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed))
        kl_loss = -0.5 * \
            tf.reduce_mean(1 + z_log_var - tf.square(z_mean) -
                           tf.exp(z_log_var))
        tf.debugging.check_numerics(
            reconstruction_loss, "Reconstruction Loss has NaN/Inf")
        tf.debugging.check_numerics(kl_loss, "KL Loss has NaN/Inf")

        # Get the latent representation z
        z = z_mean
        tf.debugging.check_numerics(z, "z (latent representation) has NaN/Inf")

        # --- Domain Loss ---
        domain_preds = domain_discriminator(z)
        domain_loss = classification_loss(d, domain_preds)
        tf.debugging.check_numerics(domain_loss, "Domain Loss has NaN/Inf")

        # --- Cycle Loss Calculations ---
        current_cycle_weight = cycle_weight(epoch)
        if current_cycle_weight > 0:  # Only compute cycle losses if weight is non-zero
            # Get hyperplane (ensure get_hyperplane is numerically stable, e.g., adds epsilon during norm)
            w, b = get_hyperplane(domain_discriminator)
            tf.debugging.check_numerics(w, "Hyperplane w has NaN/Inf")
            tf.debugging.check_numerics(b, "Hyperplane b has NaN/Inf")

            # Mirror z
            distance = tf.tensordot(z, w, axes=[[1], [0]]) + b
            tf.debugging.check_numerics(distance, "Distance has NaN/Inf")
            z_mirror = z - 2 * tf.expand_dims(distance, axis=-1) * w
            tf.debugging.check_numerics(z_mirror, "z_mirror has NaN/Inf")

            # mean matching loss
            mean_matching_loss = get_mean_matching_loss(z, z_mirror, d)
            tf.debugging.check_numerics(
                mean_matching_loss, "MMD Loss has NaN/Inf")

            # Decode mirrored z
            reconstructed_mirror = vae_model.decode(z_mirror)
            tf.debugging.check_numerics(
                reconstructed_mirror, "reconstructed_mirror has NaN/Inf")

            # Encode reconstructed mirror
            z_mirror_cycle, _, _ = vae_model.encode(reconstructed_mirror)
            tf.debugging.check_numerics(
                z_mirror_cycle, "z_mirror_cycle has NaN/Inf")

            # Domain cycle loss (calculated for metrics, not added to total loss based on previous finding)
            domain_preds_cycle = domain_discriminator(z_mirror_cycle)
            domain_cycle_loss = classification_loss(1 - d, domain_preds_cycle)
            tf.debugging.check_numerics(
                domain_cycle_loss, "Domain Cycle Loss has NaN/Inf")

            # Reflect back
            distance_cycle = tf.tensordot(
                z_mirror_cycle, w, axes=[[1], [0]]) + b
            tf.debugging.check_numerics(
                distance_cycle, "Distance_cycle has NaN/Inf")
            z_cycle = z_mirror_cycle - 2 * \
                tf.expand_dims(distance_cycle, axis=-1) * w
            tf.debugging.check_numerics(z_cycle, "z_cycle has NaN/Inf")

            # Latent cycle loss
            latent_cycle_loss = tf.reduce_mean(tf.square(z - z_cycle))
            tf.debugging.check_numerics(
                latent_cycle_loss, "Latent Cycle Loss has NaN/Inf")

            # Reconstruction cycle loss
            reconstructed_cycle = vae_model.decode(z_cycle)
            tf.debugging.check_numerics(
                reconstructed_cycle, "reconstructed_cycle has NaN/Inf")
            reconstruction_cycle_loss = tf.reduce_mean(
                tf.square(x - reconstructed_cycle))
            tf.debugging.check_numerics(
                reconstruction_cycle_loss, "Reconstruction Cycle Loss has NaN/Inf")

        else:  # If cycle weight is zero, set cycle losses to zero
            latent_cycle_loss = tf.constant(0.0)
            reconstruction_cycle_loss = tf.constant(0.0)
            # Still calculate for metrics dict consistency
            domain_cycle_loss = tf.constant(0.0)
            mean_matching_loss = tf.constant(0.0)

        # --- Total Loss ---
        total_loss = reconstruction_loss + kl_weight(epoch) * kl_loss + \
            domain_loss + \
            current_cycle_weight * \
            (latent_cycle_loss + 0.1 * reconstruction_cycle_loss) + mean_matching_loss
        tf.debugging.check_numerics(
            total_loss, "Total Loss has NaN/Inf BEFORE gradient calculation")

    # --- Gradient Calculation and Application ---
    trainable_vars = vae_model.trainable_variables + \
        domain_discriminator.trainable_variables
    grads = tape.gradient(total_loss, trainable_vars)

    # Check gradients before clipping
    for i, grad in enumerate(grads):
        if grad is not None:
            tf.debugging.check_numerics(
                grad, f"Gradient {i} for var {trainable_vars[i].name} has NaN/Inf BEFORE clipping")

    # Apply Gradient Clipping
    grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
    tf.debugging.check_numerics(
        global_norm, "Global norm of gradients has NaN/Inf")

    # Check gradients after clipping
    for i, grad in enumerate(grads):
        if grad is not None:
            tf.debugging.check_numerics(
                grad, f"Gradient {i} for var {trainable_vars[i].name} has NaN/Inf AFTER clipping")
        # else:
            # tf.print(f"Warning: Gradient {i} is None for variable {trainable_vars[i].name}") # Optional: Print if grad is None

    optimizer.apply_gradients(zip(grads, trainable_vars))

    # Return metrics
    return {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
        "domain_loss": domain_loss,
        "domain_loss_cycle": domain_cycle_loss,  # Metric only
        "reconstruction_cycle_loss": reconstruction_cycle_loss,
        "latent_cycle_loss": latent_cycle_loss,
        "mean_matching_loss": mean_matching_loss,
        "grad_global_norm": global_norm  # Add gradient norm to metrics
    }


def train_di_vae(vae_model, domain_discriminator, train_dataset, optimizer, epochs):
    """ Main training loop. """
    for epoch in range(epochs):
        # Create metrics for each loss from the train_step, including grad norm
        metric_keys = ["total_loss", "reconstruction_loss", "kl_loss", "domain_loss",
                       "domain_loss_cycle", "reconstruction_cycle_loss", "latent_cycle_loss",
                       "mean_matching_loss",  "grad_global_norm"]
        epoch_losses = {loss: tf.keras.metrics.Mean() for loss in metric_keys}

        for step, (x, d) in enumerate(train_dataset):
            # Make sure data types are correct (e.g., float32 for images)
            x = tf.cast(x, tf.float32)
            d = tf.cast(d, tf.int32)  # Assuming domain labels are integers

            try:
                losses = train_step_di(  # Use the renamed train_step function
                    vae_model, domain_discriminator, x, d, optimizer, epoch, clip_norm=1.0)  # Pass clip_norm
                for loss_key in epoch_losses:
                    # Handle potential NaN/Inf in returned losses defensively
                    loss_value = losses[loss_key]
                    if not tf.math.is_finite(loss_value):
                        print(
                            f"\nWarning: Non-finite value detected for metric '{loss_key}' in step {step}, epoch {epoch+1}. Value: {loss_value.numpy()}")
                        # Optionally: break, skip update, or use a default value
                        # For now, we'll let the Mean metric handle it (it might ignore NaNs depending on TF version)
                    epoch_losses[loss_key].update_state(loss_value)

            except tf.errors.InvalidArgumentError as e:
                # Catch errors from tf.debugging.check_numerics
                print(
                    f"\n\n!!! Numerical Instability Detected in Epoch {epoch+1}, Step {step} !!!")
                print(f"Error message: {e}")
                # Optionally: Save models, print more info, or stop training
                # Example: Print shapes
                print(f"Input shapes: x={x.shape}, d={d.shape}")
                # Trying to save model state before exiting might be useful
                print("Attempting to save models before exiting...")
                model_utils.save_model(
                    vae_model, CHECKPOINT_PATH, f'vae-error-e{epoch+1}')
                model_utils.save_model(
                    domain_discriminator, CHECKPOINT_PATH, f'domain_discriminator-error-e{epoch+1}')
                print("Models saved (if possible). Stopping training.")
                return  # Stop training

        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}:")
        log_line = []
        for key in metric_keys:
            log_line.append(f"{key}: {epoch_losses[key].result().numpy():.4f}")
        print("\t" + " | ".join(log_line))
        print("-" * 80)  # Separator line

        # Save checkpoints
        if (epoch + 1) % EPOCH_CHECKPOINT == 0 or epoch == epochs - 1:
            print(f"Saving checkpoint at epoch {epoch+1+PREVIOUS_EPOCH}...")
            model_utils.save_model(
                vae_model, CHECKPOINT_PATH, f'vae-e{epoch+1+PREVIOUS_EPOCH}')
            model_utils.save_model(
                domain_discriminator, CHECKPOINT_PATH, f'domain_discriminator-e{epoch+1+PREVIOUS_EPOCH}')
            print("Checkpoint saved.")


def linear_discriminator(input_dim, num_classes):
    """ Simple linear classifier """
    return tf.keras.Sequential([
        # Explicit Input layer is good practice
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(num_classes, activation=None)  # Logits output
    ], name="LinearDiscriminator")


if __name__ == '__main__':
    from vae import VAE, Sampling
    from load_data import DataLoader
    import init_gpu
    init_gpu.initialize_gpus()  # Initialize GPU if available

    print("Loading dataset...")
    # load combined dataset (ensure it yields (image, class_label, domain_label))
    data_loader = DataLoader(batch_size=32)  # Adjust batch size as needed
    train_dataset = data_loader.get_training_data(split='train')['combined']
    print("Dataset loaded.")

    # --- Model Initialization ---
    input_shape = (256, 256, 3)
    latent_dim = 128
    hidden_dim = 64

    if PREVIOUS_EPOCH == 0:
        print(
            f"Initializing VAE with input_shape={input_shape}, latent_dim={latent_dim}...")
        vae_model = VAE(input_shape, latent_dim, hidden_dim)

        print(
            f"Initializing Domain Discriminator with latent_dim={latent_dim}...")
        domain_discriminator = linear_discriminator(
            latent_dim, 2)  # 2 classes for domain
    else:
        print(f"Loading trained models...")
        vae_model = tf.keras.models.load_model(
            f"{CHECKPOINT_PATH}/vae-e{PREVIOUS_EPOCH}.keras", compile=False, custom_objects={'Sampling': Sampling, 'VAE': VAE})
        domain_discriminator = tf.keras.models.load_model(
            f"{CHECKPOINT_PATH}/domain_discriminator-e{PREVIOUS_EPOCH}.keras", compile=False)

    # Build the VAE model by calling it once (helps with saving/loading)
    # Use tf.data.Dataset.take(1) to get one batch, then next(iter(...))
    sample_batch = next(iter(train_dataset.take(1)))
    sample_input = sample_batch[0]  # Get the image tensor 'x'
    _ = vae_model(sample_input)  # Build the model
    vae_model.summary()  # Print VAE summary

    # Build the discriminator
    sample_latent = tf.random.normal(
        (sample_input.shape[0], latent_dim))  # Match batch size
    _ = domain_discriminator(sample_latent)
    domain_discriminator.summary()  # Print discriminator summary

    # --- Optimizer ---
    learning_rate = 1e-5  # Try an even lower learning rate
    print(f"Using Adam optimizer with learning_rate={learning_rate}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # --- Training ---
    epochs = 2000
    print(f"Starting training for {epochs} epochs...")
    train_di_vae(vae_model, domain_discriminator,
                 train_dataset, optimizer, epochs=epochs)
    print("Training finished.")

    # --- Final Model Saving ---
    print("Saving final models...")
    model_utils.save_model(vae_model, SAVE_PATH,
                           f'vae-e{epochs+PREVIOUS_EPOCH}')
    model_utils.save_model(domain_discriminator, SAVE_PATH,
                           f'domain_discriminator-e{epochs+PREVIOUS_EPOCH}')
    print("Final models saved.")
