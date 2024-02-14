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

if __name__ == '__main__':
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

    neutrophil_index = label_map['neutrophil']
    basophil_index = label_map['basophil']
    monocyte_index = label_map['monocyte']

    # Create a mask for selecting only the desired classes
    mask = (all_labels_array == neutrophil_index) | \
           (all_labels_array == basophil_index) | \
           (all_labels_array == monocyte_index)

    # Apply the mask to filter both latent data and labels
    filtered_latent_data = latent_data_reshaped[mask]
    filtered_labels = all_labels_array[mask]

    # UMAP for latent space
    latent_data_umap = UMAP(n_neighbors=13, min_dist=0.1, n_components=2, metric='euclidean').fit_transform(
        latent_data_reshaped)

    plt.figure(figsize=(12, 10), dpi=150)
    scatter = plt.scatter(latent_data_umap[:, 0], latent_data_umap[:, 1], s=1, c=all_labels_array, cmap='plasma')

    color_map = plt.cm.plasma(np.linspace(0, 1, len(set(all_labels_array))))
    class_names = [inverse_label_map[i] for i in range(len(inverse_label_map))]

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[i],
                                 markerfacecolor=color_map[i], markersize=10) for i in range(len(class_names))]
    plt.legend(handles=legend_handles, loc='lower right', title='Cell Types')

    plt.title(f'Latent Space Representation - (Epoch {epoch})', fontsize=18)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)

    umap_figure_filename = os.path.join(umap_dir, f'umap_epoch_{epoch}.png')

    # Save the UMAP figure
    plt.savefig(umap_figure_filename, dpi=300)
    print("it is saved")
    plt.close()

"""

    geodesic_ab_fisher = normal.metric.geodesic(random_myeloblast_point, random_neutrophil_banded_point)
    n_points = 20
    t = gs.linspace(0, 1, n_points)
    pdfs = normal.point_to_pdf(geodesic_ab_fisher(t))
    x = gs.linspace(-5.0, 25.0, 100)

    fig = plt.figure(figsize=(10, 5))
    cc = gs.zeros((n_points, 3))
    cc[:, 2] = gs.linspace(0, 1, n_points)
    for i in range(n_points):
        plt.plot(x, pdfs(x)[i, :], color=cc[i, :])
    plt.title("Corresponding interpolation between pdfs");
    pdf_figure_filename = os.path.join(pdf_dir, f'pdf_interpolation_epoch_{epoch}.png')
    plt.savefig(pdf_figure_filename)
    plt.close(fig)
"""
