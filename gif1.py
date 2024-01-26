from scipy.stats import stats
from torch.utils.data import DataLoader
import torch
from umap import UMAP
from Dataloader_4 import Dataloader, label_map
from model4 import VariationalAutoencodermodel4, reparametrize
import os
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import geomstats.backend as gs
import matplotlib.pyplot as plt
from Beta_Visualization import Beta
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.geometry.connection import Connection
from geomstats.geometry.complex_manifold import ComplexManifold
from sklearn.preprocessing import MinMaxScaler

import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean


import matplotlib
import matplotlib.pyplot as plt
from geomstats.information_geometry.normal import NormalDistributions

normal = NormalDistributions(sample_dim=1)


beta = BetaDistributions()
beta_p = Beta()


label_map = {
    'basophil': 0,
    'eosinophil': 1,
    'erythroblast': 2,
    'myeloblast': 3,
    'promyelocyte': 4,
    'myelocyte': 5,
    'metamyelocyte': 6,
    'neutrophil_banded': 7,
    'neutrophil_segmented': 8,
    'monocyte': 9,
    'lymphocyte_typical': 10,
    'lymphocyte_atypical': 11,
    'smudge_cell': 12,
}

"""


def plot_reconstructed(vae_model, dim1=0, dim2=3, r0=(-3, 3), r1=(-3, 3), n=12, latent_dim=30):
    w = 128
    img = np.zeros((n*w, n*w, 3))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.zeros((1, latent_dim)).to(device)
            z[0, dim1] = x
            z[0, dim2] = y
            x_hat = vae_model.decoder(z)
            x_hat = vae_model.img_decoder(x_hat)
            x_hat = x_hat.reshape(3, w, w).permute(1, 2, 0).to('cpu').detach().numpy()  # Reshape and transpose
            img[(n - 1 - i) * w:(n - 1 - i + 1) * w, j * w:(j + 1) * w, :] = x_hat  # Place the image in the grid
    plt.imshow(img, extent=[*r0, *r1])
    plt.axis('off')
    plt.savefig('reconstructed_images.png', bbox_inches='tight', pad_inches=0)
    
plot_reconstructed(model, dim1=0, dim2=1, r0=(-3, 3), r1=(-3, 3), n=12, latent_dim=30)    
"""


if __name__ == '__main__':
    inverse_label_map = {v: k for k, v in label_map.items()}  # inverse mapping for UMAP
    batch_size = 128
    num_classes = len(label_map)
    epoch = 140

    histo_dir = 'histo_path'
    if not os.path.exists(histo_dir):
        os.makedirs(histo_dir)

    q_dir = 'q_path'
    if not os.path.exists(q_dir):
        os.makedirs(q_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VariationalAutoencodermodel4(latent_dim=30)
    model_save_path = 'trained_model4cp2_new5.pth'
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    model.eval()

    train_dataset = Dataloader(split='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=1)

    # Load all latent representations
    latent_dir = 'latent_data4cp2_new5'
    latents_path = os.path.join(latent_dir, f'latent_epoch_{epoch}.npy')
    label_dir = 'label_data4cp2_new5'
    labels_path = os.path.join(label_dir, f'label_epoch_{epoch}.npy')

    # Load all latent representations
    latent_data = np.load(latents_path)
    latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
    print("Latent data shape:", latent_data.shape)

    # Load all labels
    all_labels_array = np.load(labels_path)
    print("Labels array shape:", all_labels_array.shape)

    # print("Labels array shape:", all_labels_array.shape)

    # Filter out the 'erythroblast' class
    erythroblast_class_index = label_map['erythroblast']
    mask = all_labels_array != erythroblast_class_index
    filtered_latent_data = latent_data[mask]
    filtered_labels = all_labels_array[mask]

    plt.hist(latent_data.flatten(), bins=30, density=True, alpha=0.6, color='g')
    plt.title("Histogram of Latent Data")
    histo_filename = os.path.join(histo_dir, f'histo_epoch_{epoch}.png')
    plt.savefig("latent_data_histogram.png")  # Save histogram
    plt.close()

    # Q-Q plot
    stats.probplot(latent_data.flatten(), dist="norm", plot=plt)
    plt.title("Q-Q Plot of Latent Data")
    q_filename = os.path.join(q_dir, f'histo_epoch_{epoch}.png')
    plt.savefig("latent_data_qqplot.png")  # Save Q-Q plot
    plt.close()

    # Shapiro-Wilk Test
    shapiro_test = stats.shapiro(latent_data.flatten())
    print("Shapiro-Wilk Test: ", shapiro_test)

    # Kolmogorov-Smirnov Test
    ks_test = stats.kstest(latent_data.flatten(), 'norm',
                           args=(latent_data.mean(), latent_data.std()))
    print("Kolmogorov-Smirnov Test: ", ks_test)

    mean = np.mean(latent_data, axis=0)
    covariance_matrix = np.cov(latent_data, rowvar=False)

    # Check for centered distribution
    is_centered = np.allclose(mean, 0)

    # Check for diagonal distribution
    is_diagonal = np.allclose(covariance_matrix, np.diag(np.diagonal(covariance_matrix)))

    # Determine distribution type
    if is_centered:
        distribution_type = 'centered'
    elif is_diagonal:
        distribution_type = 'diagonal'
    else:
        distribution_type = 'general'

    print("Distribution type:", distribution_type)


