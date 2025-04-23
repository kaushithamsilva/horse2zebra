import os
import tensorflow as tf


class DataLoader:
    """Class to load and preprocess images for training a cross domain synthesis model."""

    def __init__(self, data_path="../../dataset/", batch_size=32):
        self.data_path = data_path
        self.batch_size = batch_size

    def load_images(self, folder_name):
        """"Load images from the specified folder and normalize them."""
        path_name = os.path.join(self.data_path, folder_name)
        print(f"Loading images from {path_name}...")
        if not os.path.exists(path_name):
            raise FileNotFoundError(f"Data folder {path_name} does not exist.")
        dataset = tf.keras.utils.image_dataset_from_directory(
            path_name,
            label_mode=None,  # No labels since this is unsupervised
            image_size=(256, 256),  # Resize images to 256x256
            batch_size=self.batch_size  # Batch size
        )
        return self._normalize_dataset(dataset)  # Normalize images to [0, 1]

    def _normalize_dataset(self, dataset):
        return dataset.map(lambda x: x / 255.0)

    def get_training_data(self, split='train'):
        """Load and combine normalized data for the specified split (train or test)."""
        horse_dataset = self.load_images(f"{split}A")
        zebra_dataset = self.load_images(f"{split}B")

        # Create labels for the datasets: 0 for horse, 1 for zebra
        horse_labels = tf.data.Dataset.from_tensor_slices(
            tf.zeros(horse_dataset.cardinality(), dtype=tf.int32))
        zebra_labels = tf.data.Dataset.from_tensor_slices(
            tf.ones(zebra_dataset.cardinality(), dtype=tf.int32))

        # Combine datasets and labels
        horse_data = tf.data.Dataset.zip((horse_dataset, horse_labels))
        zebra_data = tf.data.Dataset.zip((zebra_dataset, zebra_labels))

        # Concatenate horse and zebra datasets
        combined_dataset = horse_data.concatenate(zebra_data)

        return combined_dataset.shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE)
