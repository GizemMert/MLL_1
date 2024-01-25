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

    # Filter out the 'erythroblast' class
    erythroblast_class_index = label_map['erythroblast']
    mask = all_labels_array != erythroblast_class_index
    filtered_latent_data = latent_data_reshaped[mask]
    filtered_labels = all_labels_array[mask]

    # UMAP for latent space
    latent_data_umap = UMAP(n_neighbors=13, min_dist=0.1, n_components=2, metric='euclidean').fit_transform(
        filtered_latent_data)

    myeloblast_umap_points = latent_data_umap[filtered_labels == label_map['myeloblast']]
    neutrophil_banded_umap_points = latent_data_umap[filtered_labels == label_map['neutrophil_banded']]

    random_myeloblast_point = myeloblast_umap_points[np.random.choice(myeloblast_umap_points.shape[0])]
    random_neutrophil_banded_point = neutrophil_banded_umap_points[
        np.random.choice(neutrophil_banded_umap_points.shape[0])]

    reshaped_myeloblast_point = random_myeloblast_point.reshape(1, 2)
    reshaped_neutrophil_banded_point = random_neutrophil_banded_point.reshape(1, 2)

    """
    fig = plt.figure(figsize=(12, 10), dpi=150)
    ax = fig.add_subplot(111)
    # scatter = ax.scatter(latent_data_umap[:, 0], latent_data_umap[:, 1], s=100, c=filtered_labels, cmap='Spectral')
    cc = gs.zeros((20, 3))
    cc[:, 2] = gs.linspace(0, 1, 20)
    print("Random Myeloblast Point:", random_myeloblast_point)
    print("Type of Random Myeloblast Point:", type(random_myeloblast_point))
    print("Shape of Random Myeloblast Point:", random_myeloblast_point.shape)

    print("Random Neutrophil Banded Point:", random_neutrophil_banded_point)
    print("Type of Random Neutrophil Banded Point:", type(random_neutrophil_banded_point))
    print("Shape of Random Neutrophil Banded Point:", random_neutrophil_banded_point.shape)

    if random_myeloblast_point is not None and random_neutrophil_banded_point is not None:
        if not (np.isnan(random_myeloblast_point).any() or np.isnan(random_neutrophil_banded_point).any()):
            if random_myeloblast_point.shape == (2,) and random_neutrophil_banded_point.shape == (2,):
                t = np.linspace(0, 1, 50)
                geod = beta.metric.geodesic(initial_point=random_myeloblast_point,
                                            end_point=random_neutrophil_banded_point)(t)
                points_x = geod[:, 0]
                points_y = geod[:, 1]
                ax.scatter(points_x, points_y, s=50, c='black', marker='o')

                # Print the geodesic points
                print("Geodesic points:")
                print(geod)
            else:
                print("The shape of one or both points is incorrect.")
        else:
            print("One of the points contains NaN values.")
    else:
        print("One of the points is None.")

    x_min, x_max = latent_data_umap[:, 0].min(), latent_data_umap[:, 0].max()
    y_min, y_max = latent_data_umap[:, 1].min(), latent_data_umap[:, 1].max()

    initial_point_color = 'blue'
    end_point_color = 'orange'
    ax.scatter(random_myeloblast_point[0], random_myeloblast_point[1], s=100, c=initial_point_color, marker='o',
               label='Myeloblast')
    ax.scatter(random_neutrophil_banded_point[0], random_neutrophil_banded_point[1], s=100, c=end_point_color,
               marker='o', label='Neutrophil Banded')

    # Create a custom legend
    initial_patch = mpatches.Patch(color=initial_point_color, label='Myeloblast')
    end_patch = mpatches.Patch(color=end_point_color, label='Neutrophil Banded')
    ax.legend(handles=[initial_patch, end_patch])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f'Geodesic Plot - (Epoch {epoch})', fontsize=18)
    ax.set_xlabel('UMAP Dimension 1', fontsize=14)
    ax.set_ylabel('UMAP Dimension 2', fontsize=14)
    umap_figure_filename = os.path.join(umap_dir, f'geodesic_plot.png_{epoch}.png')
    plt.savefig(umap_figure_filename, bbox_inches='tight', dpi=300)
    plt.close()

"""

    geodesic_ab_fisher = normal.metric.geodesic(random_myeloblast_point, random_neutrophil_banded_point)
    n_points = 20
    t = gs.linspace(0, 1, n_points)
    pdfs = normal.point_to_pdf(geodesic_ab_fisher(t))
    x = gs.linspace(-5.0, 15.0, 100)

    fig = plt.figure(figsize=(10, 5))
    cc = gs.zeros((n_points, 3))
    cc[:, 2] = gs.linspace(0, 1, n_points)
    for i in range(n_points):
        plt.plot(x, pdfs(x)[i, :], color=cc[i, :])
    plt.title("Corresponding interpolation between pdfs");
    pdf_figure_filename = os.path.join(pdf_dir, f'pdf_interpolation_epoch_{epoch}.png')
    plt.savefig(pdf_figure_filename)
    plt.close(fig)
