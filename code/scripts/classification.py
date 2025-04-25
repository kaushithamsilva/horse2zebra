# classification.py
# Script to evaluate a linear latent classifier on the horse↔zebra test dataset

import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Import your DataLoader, VAE, and linear classifier factory
from load_data import DataLoader
from vae import VAE
from vae import VAE, Sampling

# === Configuration: set these paths and parameters manually ===
DATA_PATH = "../../dataset/"                 # Path to the root dataset folder
BATCH_SIZE = 32                                # Batch size for evaluation
SAVE_PATH = '../models/di_vae/'
# Path to linear_discriminator weights (e.g. .ckpt)
CLASSIFIER_WEIGHTS_PATH = "path/to/cls_weights"
LATENT_DIM = 128                               # Must match the VAE latent dimension


def evaluate_classifier():
    # Prepare DataLoader and load test split
    loader = DataLoader(data_path=DATA_PATH, batch_size=BATCH_SIZE)
    test_ds = loader.get_training_data(split='test')['combined']

    # Load the pretrained VAE (without compiling)
    epochs = 750
    vae = tf.keras.models.load_model(
        f"{SAVE_PATH}/vae-e{epochs}.keras", compile=False, custom_objects={'Sampling': Sampling, 'VAE': VAE})

    print("Loaded VAE:")
    vae.summary()

    classifier = tf.keras.models.load_model(
        f"{SAVE_PATH}/domain_discriminator-e{epochs}.keras", compile=False)
    print("Loaded linear classifier:")
    classifier.summary()

    # Accumulate predictions and true labels
    all_preds = []
    all_trues = []

    for x_batch, y_batch in test_ds:
        # Ensure correct dtype
        x_batch = tf.cast(x_batch, tf.float32)
        y_batch = tf.cast(y_batch, tf.int32)

        # Encode to latent mean
        z_mean, _, _ = vae.encode(x_batch)

        # Predict domain logits
        logits = classifier(z_mean)
        preds = tf.argmax(logits, axis=-1).numpy()

        all_preds.append(preds)
        all_trues.append(y_batch.numpy())

    # Concatenate all batches
    y_preds = np.concatenate(all_preds)
    y_trues = np.concatenate(all_trues)

    # Print metrics
    print("\nClassification Report:")
    print(classification_report(y_trues, y_preds,
          target_names=['horse', 'zebra']))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_trues, y_preds)
    print(cm)


if __name__ == '__main__':
    evaluate_classifier()
