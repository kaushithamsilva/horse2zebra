"""
Synthesize a sample of the dataset using the trained model.
"""
import tensorflow as tf
import model_utils
from vae import VAE, Sampling
import init_gpu
from load_data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from hyperplane import Hyperplane

SAVE_PATH = '../models/di_vae/'
CHECKPOINT_PATH = SAVE_PATH + 'checkpoints'

if __name__ == "__main__":
    init_gpu.initialize_gpus()

    # load models
    epochs = 450
    vae_model = tf.keras.models.load_model(
        f"{CHECKPOINT_PATH}/vae-e{epochs}.keras", compile=False, custom_objects={'Sampling': Sampling, 'VAE': VAE})
    domain_discriminator = tf.keras.models.load_model(
        f"{CHECKPOINT_PATH}/domain_discriminator-e{epochs}.keras", compile=False)

    # load data
    data_loader = DataLoader()
    dataset = data_loader.get_training_data()
    horse_dataset = dataset['horse']
    zebra_dataset = dataset['zebra']

    # load a batch from zebra dataset
    zebra_batch = zebra_dataset.take(1)
    zebra_batch = next(iter(zebra_batch))

    horse_batch = horse_dataset.take(1)
    horse_batch = next(iter(horse_batch))

    # initialize hyperplane
    domain_hyperplane = Hyperplane(domain_discriminator)

    # synthesize a zebra image from a horse image
    x_batch, y_batch = horse_batch
    index = np.random.randint(0, len(x_batch), 1)[0]

    # reconstruct the image using the VAE model
    reconstructed, _, _ = vae_model(x_batch[index:index+1])

    # get the latent representation of the horse image
    _, _, z_horse = vae_model.encode(x_batch[index:index+1])
    z_zebra = domain_hyperplane.get_mirror_image(z_horse[0])

    # reconstruct the image using the VAE model
    synthesized = vae_model.decode(z_zebra.numpy().reshape(1, -1))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title("Orginal Hose Image")
    plt.imshow(x_batch[index])

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title("Reconstructed Horse Image")
    plt.imshow(reconstructed.numpy().reshape(256, 256, 3))

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title("Synthesized Zebra Image")
    plt.imshow(synthesized.numpy().reshape(256, 256, 3))

    plt.savefig(f"synthesized_zebra-e{epochs}.png", bbox_inches='tight')
