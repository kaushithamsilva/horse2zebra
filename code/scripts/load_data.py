import os
import tensorflow as tf


class DataLoader:
    """Class to load and preprocess images for training a cross domain synthesis model."""

    def __init__(self, data_path="../../dataset/", batch_size=32):
        self.data_path = data_path
        self.batch_size = batch_size

    def get_training_data(self, split='train'):
        """Load and combine normalized data for the specified split (train or test)."""
        # Load the image datasets without labels first
        horse_dataset = tf.keras.utils.image_dataset_from_directory(
            os.path.join(self.data_path, f"{split}A"),
            label_mode=None,  # No labels from directory structure
            image_size=(256, 256),
            batch_size=self.batch_size
        )

        zebra_dataset = tf.keras.utils.image_dataset_from_directory(
            os.path.join(self.data_path, f"{split}B"),
            label_mode=None,  # No labels from directory structure
            image_size=(256, 256),
            batch_size=self.batch_size
        )

        # Normalize the images
        horse_dataset = horse_dataset.map(lambda x: x / 255.0)
        zebra_dataset = zebra_dataset.map(lambda x: x / 255.0)

        # Create domain labels manually after loading
        # Get number of batches to create labels for each batch
        horse_batches = tf.data.experimental.cardinality(horse_dataset).numpy()
        if horse_batches < 0:  # If cardinality is unknown
            print("Warning: Horse dataset cardinality unknown, using estimate")
            horse_batches = 100  # Use a reasonable estimate or count files manually

        zebra_batches = tf.data.experimental.cardinality(zebra_dataset).numpy()
        if zebra_batches < 0:  # If cardinality is unknown
            print("Warning: Zebra dataset cardinality unknown, using estimate")
            zebra_batches = 100  # Use a reasonable estimate or count files manually

        # Create domain labels for each batch (0 for horse, 1 for zebra)
        def add_horse_labels(images):
            batch_size = tf.shape(images)[0]
            return images, tf.zeros(batch_size, dtype=tf.int32)

        def add_zebra_labels(images):
            batch_size = tf.shape(images)[0]
            return images, tf.ones(batch_size, dtype=tf.int32)

        # Add domain labels to each batch
        horse_with_labels = horse_dataset.map(add_horse_labels)
        zebra_with_labels = zebra_dataset.map(add_zebra_labels)

        # Keep these separate for evaluation
        return {
            'horse': horse_with_labels,
            'zebra': zebra_with_labels,
            'combined': horse_with_labels.concatenate(zebra_with_labels).prefetch(tf.data.AUTOTUNE)
        }
