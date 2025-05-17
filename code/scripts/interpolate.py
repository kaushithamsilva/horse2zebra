import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from vae import VAE, Sampling
from hyperplane import Hyperplane
from load_data import DataLoader
import init_gpu

# --- Configuration ---
DATA_PATH = "../../dataset/"
CHECKPOINT_PATH = '../../models/gan_di_vae/checkpoints'
EPOCH = 5000
LATENT_DIM = 128
N_STEPS = 10

# Initialize GPU
init_gpu.initialize_gpus()

# Load trained VAE
vae = tf.keras.models.load_model(
    f"{CHECKPOINT_PATH}/vae-e{EPOCH}.keras",
    compile=False,
    custom_objects={'Sampling': Sampling, 'VAE': VAE}
)

# Prepare dataset
loader = DataLoader(data_path=DATA_PATH, batch_size=32)
horse_ds = loader.get_training_data(split='train')['horse']

# Grab one batch of horse images
images, _ = next(iter(horse_ds))  # images: (batch, 256,256,3)
images = images.numpy()
batch_size = images.shape[0]

# Randomly select two distinct horses
idx = np.random.choice(batch_size, size=2, replace=False)
img1, img2 = images[idx[0]], images[idx[1]]

# Encode to latent means
z1_mean, _, _ = vae.encode(img1[None])
z2_mean, _, _ = vae.encode(img2[None])

# Interpolate
alphas = np.linspace(0.0, 1.0, N_STEPS)
zs = [(1 - a) * z1_mean + a * z2_mean for a in alphas]

# Decode all
decoded = [vae.decode(z).numpy().reshape(256, 256, 3) for z in zs]

# Plot grid
fig, axes = plt.subplots(2, N_STEPS//2, figsize=(15, 6))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(decoded[i])
    ax.axis('off')
plt.tight_layout()
plt.savefig(f"interpolation_e{EPOCH}.png", bbox_inches='tight')
print(f"Saved interpolation grid as interpolation_e{EPOCH}.png")
