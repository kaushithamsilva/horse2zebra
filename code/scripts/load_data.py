import os
import tensorflow as tf


class DataLoader:
    """Class to load and preprocess images for training a cross domain synthesis model."""

    def __init__(self, data_path="../../dataset/", batch_size=32):
        self.data_path = data_path
        self.batch_size = batch_size

    def get_training_data(self, split='train'):
        """Load and combine normalized data for the specified split (train or test)."""

        # Load images from directories
        horse_dir = os.path.join(self.data_path, f"{split}A")
        zebra_dir = os.path.join(self.data_path, f"{split}B")

        # Count files to create appropriate label arrays
        horse_files = len([f for f in os.listdir(horse_dir)
                          if os.path.isfile(os.path.join(horse_dir, f))])
        zebra_files = len([f for f in os.listdir(zebra_dir)
                          if os.path.isfile(os.path.join(zebra_dir, f))])

        # Create datasets with labels directly
        horse_dataset = tf.keras.utils.image_dataset_from_directory(
            horse_dir,
            labels=tf.zeros(horse_files, dtype=tf.int32),
            label_mode="int",
            image_size=(256, 256),
            batch_size=self.batch_size
        )

        zebra_dataset = tf.keras.utils.image_dataset_from_directory(
            zebra_dir,
            labels=tf.ones(zebra_files, dtype=tf.int32),
            label_mode="int",
            image_size=(256, 256),
            batch_size=self.batch_size
        )

        # Normalize images
        horse_dataset = horse_dataset.map(lambda x, y: (x / 255.0, y))
        zebra_dataset = zebra_dataset.map(lambda x, y: (x / 255.0, y))

        # Concatenate datasets
        combined_dataset = horse_dataset.concatenate(zebra_dataset)
        return combined_dataset.shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE)
