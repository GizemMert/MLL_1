from torch.utils.data import DataLoader
import torch
from umap import UMAP
from Dataloader_4 import Dataloader, label_map
from model4 import VariationalAutoencodermodel4, reparametrize
import os
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import geomstats.backend as gs
import matplotlib.pyplot as plt
from Beta_Visualization import Beta
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.geometry.connection import Connection
from geomstats.geometry.complex_manifold import ComplexManifold

import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean

import matplotlib
import matplotlib.pyplot as plt


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
    neutrophil_banded_umap_points = latent_data_umap[filtered_labels == label_map['lymphocyte_typical']]

    random_myeloblast_point = myeloblast_umap_points[np.random.choice(myeloblast_umap_points.shape[0])]
    random_neutrophil_banded_point = neutrophil_banded_umap_points[
        np.random.choice(neutrophil_banded_umap_points.shape[0])]

    reshaped_myeloblast_point = random_myeloblast_point.reshape(1, 2)
    reshaped_neutrophil_banded_point = random_neutrophil_banded_point.reshape(1, 2)



    fig = plt.figure(figsize=(12, 10), dpi=150)
    g_s = GridSpec(1, 2, width_ratios=[4, 1], figure=fig)
    ax = fig.add_subplot(g_s[0])
    scatter = ax.scatter(latent_data_umap[:, 0], latent_data_umap[:, 1], s=100, c=filtered_labels, cmap='Spectral')
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
                beta_p.plot_geodesic(ax=ax, initial_point=reshaped_myeloblast_point, end_point=reshaped_neutrophil_banded_point, n_points=20)
            else:
                print("The shape of one or both points is incorrect.")
        else:
            print("One of the points contains NaN values.")
    else:
        print("One of the points is None.")
    ax.set_aspect('equal')

    x_min, x_max = np.min(latent_data_umap[:, 0]), np.max(latent_data_umap[:, 0])
    y_min, y_max = np.min(latent_data_umap[:, 1]), np.max(latent_data_umap[:, 1])

    zoom_factor = 0.40  # Smaller values mean more zoom
    padding_factor = 0.3  # Adjust padding around the zoomed area

    # Calculate the range for zooming in based on the zoom factor
    x_range = (x_max - x_min) * zoom_factor
    y_range = (y_max - y_min) * zoom_factor

    # Calculate the center of the data
    center_x = (x_max + x_min) / 2
    center_y = (y_max + y_min) / 2

    # Calculate new limits around the center of the data
    new_x_min = center_x - (x_range * (1 + padding_factor))
    new_x_max = center_x + (x_range * (1 + padding_factor))
    new_y_min = center_y - (y_range * (1 + padding_factor))
    new_y_max = center_y + (y_range * (1 + padding_factor))

    # Apply the new limits to zoom in on the plot
    ax.set_xlim(new_x_min, new_x_max)
    ax.set_ylim(new_y_min, new_y_max)

    ax.set_title(f'Latent Space Representation - (Epoch {epoch})', fontsize=18)
    ax.set_xlabel('UMAP Dimension 1', fontsize=16)
    ax.set_ylabel('UMAP Dimension 2', fontsize=16)

    # Second subplot for the legend
    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis('off')  # Turn off the axis for the legend subplot

    unique_filtered_labels = np.unique(filtered_labels)
    filtered_class_names = [inverse_label_map[label] for label in unique_filtered_labels if label in inverse_label_map]
    color_map = plt.cm.Spectral(np.linspace(0, 1, len(unique_filtered_labels)))

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=filtered_class_names[i],
                                 markerfacecolor=color_map[i], markersize=18) for i in range(len(filtered_class_names))]

    ax_legend.legend(handles=legend_handles, loc='center', fontsize=16, title='Cell Types')

    plt.tight_layout()
    umap_figure_filename = os.path.join(umap_dir, f'umap_epoch_{epoch}.png')
    plt.savefig(umap_figure_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

    n_points = 20
    t = gs.linspace(0, 1, n_points)

    pdfs = beta.point_to_pdf(beta.metric.geodesic(random_myeloblast_point, random_neutrophil_banded_point)(t))
    x = gs.linspace(0.0, 1.0, 100)

    fig = plt.figure(figsize=(10, 5))
    cc = gs.zeros((n_points, 3))
    cc[:, 2] = gs.linspace(0, 1, n_points)
    for i in range(n_points):
        plt.plot(x, pdfs(x)[:, i], color=cc[i, :])
    plt.title("Corresponding interpolation between pdfs")
    pdf_figure_filename = os.path.join(pdf_dir, f'pdf_interpolation_epoch_{epoch}.png')
    plt.savefig(pdf_figure_filename)
    plt.close(fig)
