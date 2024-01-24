from torch.utils.data import DataLoader
import torch
from umap import UMAP
from Dataloader_4 import Dataloader, label_map
from model4 import VariationalAutoencodermodel4, reparametrize
import os
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import geomstats._backend as gs
import matplotlib.pyplot as plt
from Beta_Visualization import Beta
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.geometry.connection import Connection
from geomstats.geometry.complex_manifold import ComplexManifold
import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean
from Umap_new import latent_data_umap
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.information_geometry.normal import NormalDistributions


if __name__ == '__main__':
    normal = NormalDistributions(sample_dim=1)

    complex_space = ComplexManifold(dim=2)

    beta = Connection(space=complex_space)

    beta_p = Beta()

    beta_dir = 'beta_manifold_path'
    if not os.path.exists(beta_dir):
        os.makedirs(beta_dir)

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

    inverse_label_map = {v: k for k, v in label_map.items()}  # inverse mapping for UMAP
    batch_size = 128
    num_classes = len(label_map)
    epoch = 140

    umap_dir = 'umap_manifold_path'
    if not os.path.exists(umap_dir):
        os.makedirs(umap_dir)

    pdf_dir = 'pdf_manifold_path'
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)

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
    print("Latent data shape:", latent_data_reshaped.shape)

    # Load all labels
    all_labels_array = np.load(labels_path)
    print("Labels array shape:", all_labels_array.shape)

    # print("Labels array shape:", all_labels_array.shape)

    # Filter out the 'erythroblast' class
    erythroblast_class_index = label_map['erythroblast']
    mask = all_labels_array != erythroblast_class_index
    filtered_latent_data = latent_data_reshaped[mask]
    filtered_labels = all_labels_array[mask]

    myeloblast_umap_points = latent_data_umap[filtered_labels == label_map['myeloblast']]
    neutrophil_banded_umap_points = latent_data_umap[filtered_labels == label_map['lymphocyte_typical']]

    random_myeloblast_point = myeloblast_umap_points[np.random.choice(myeloblast_umap_points.shape[0])]
    random_neutrophil_banded_point = neutrophil_banded_umap_points[
        np.random.choice(neutrophil_banded_umap_points.shape[0])]

    geodesic_ab_fisher = normal.metric.geodesic(random_myeloblast_point, random_neutrophil_banded_point)

    n_points = 20
    t = gs.linspace(0, 1, n_points)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    cc = gs.zeros((n_points, 3))
    cc[:, 2] = gs.linspace(0, 1, n_points)

    visualization.plot(
        geodesic_ab_fisher(t),
        ax=ax,
        space="H2_poincare_half_plane",
        label="point on geodesic",
        color=cc,
    )

    ax.set_xlim(0.0, 15.0)
    ax.set_ylim(0.0, 5.0)
    ax.set_title("Geodesic between two normal distributions for the Fisher-Rao metric")

    pdf_figure_filename = os.path.join(beta_dir, f'beta_interpolation_epoch_{epoch}.png')
    plt.savefig(pdf_figure_filename)
    plt.close(fig)