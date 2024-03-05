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
# import geomstats.backend as gs
import matplotlib.pyplot as plt
from Beta_Visualization import Beta
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.geometry.connection import Connection
from geomstats.geometry.complex_manifold import ComplexManifold
from sklearn.preprocessing import MinMaxScaler

import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean
inverse_label_map = {v: k for k, v in label_map.items()}  # inverse mapping for UMAP

import matplotlib
import matplotlib.pyplot as plt
from geomstats.information_geometry.normal import NormalDistributions


if __name__ == '__main__':

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

    umap_dir = 'umap_comparison_img_gene'
    if not os.path.exists(umap_dir):
        os.makedirs(umap_dir)

    epoch = 150

    latent_dir = 'latent_data4cp2_new5_std_gen_2'
    latents_path = os.path.join(latent_dir, f'latent_epoch_{epoch}.npy')
    label_dir = 'label_data4cp2_new5_std_gen_2'
    labels_path = os.path.join(label_dir, f'label_epoch_151.npy')
    latent_data = np.load(latents_path)
    # latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
    print("Latent data shape:", latent_data.shape)
    all_labels_array = np.load(labels_path)
    print("Labels array shape:", all_labels_array.shape)


    # gene latents
    z_dir_ge = 'z_variables_GE_3'
    epoch_of_gen = 290

    class_labels_gen = [0, 1, 2, 3]
    # monocyte : class 1
    # myeloblast : class 2
    # basophil : class 0
    # neutrophil : class 3

    class_label_n = 3
    class_label_m = 1
    class_label_myeloid = 2

    z_filename_n = os.path.join(z_dir_ge, f'class_{class_label_n}_z_epoch_{epoch_of_gen}.npy')
    z_filename_myeloid = os.path.join(z_dir_ge, f'class_{class_label_myeloid}_z_epoch_{epoch_of_gen}.npy')
    ref_z_class_neutrophils = np.load(z_filename_n)
    ref_z_class_myeloid = np.load(z_filename_myeloid)
    z_filename_m = os.path.join(z_dir_ge, f'class_{class_label_m}_z_epoch_{epoch_of_gen}.npy')
    ref_z_class_mono = np.load(z_filename_m)

    latent_dir_ge_2 = 'latent_variables_GE_3_cp'
    z_dir_ge_2 = 'z_variables_GE_3_cp'
    class_label_n_blood = 2
    class_label_n_lung = 5

    z_filename_n_blood = os.path.join(z_dir_ge_2, f'class_{class_label_n_blood}_z_epoch_{epoch_of_gen}.npy')
    ref_z_class_n_blood = np.load(z_filename_n_blood)

    z_filename_n_lung = os.path.join(z_dir_ge_2, f'class_{class_label_n_lung}_z_epoch_{epoch_of_gen}.npy')
    ref_z_class_n_lung = np.load(z_filename_n_lung)


    # Filter out the 'erythroblast' class
    erythroblast_class_index = label_map['erythroblast']
    mask = all_labels_array != erythroblast_class_index
    filtered_latent_data = latent_data[mask]
    filtered_labels = all_labels_array[mask]

    gene_data_sizes = [
        ref_z_class_neutrophils.shape[0],
        ref_z_class_myeloid.shape[0],
        ref_z_class_mono.shape[0],
        ref_z_class_n_blood.shape[0],
        ref_z_class_n_lung.shape[0]
    ]


    gene_labels = np.concatenate([
        np.full(gene_data_sizes[0], "neutrophil gene"),
        np.full(gene_data_sizes[1], "myeloid gene"),
        np.full(gene_data_sizes[2], "monocyte gene"),
        np.full(gene_data_sizes[3], "lung neutrophil gene"),
        np.full(gene_data_sizes[4], "blood neutrophil gene"),
    ])

    gene_data_labels = gene_labels

    # UMAP for latent space

    reducer = UMAP(n_neighbors=13, min_dist=1, n_components=2, metric='euclidean')
    reducer.fit(filtered_latent_data)
    latent_data_transformed = reducer.transform(filtered_latent_data)

    combined_gene_data = np.concatenate([
        ref_z_class_neutrophils,
        ref_z_class_myeloid,
        ref_z_class_mono,
        ref_z_class_n_blood,
        ref_z_class_n_lung], axis=0)

    combined_gene_data_transformed = reducer.transform(combined_gene_data)


    fig = plt.figure(figsize=(12, 10), dpi=150)
    gs = GridSpec(1, 2, width_ratios=[4, 1], figure=fig)

    ax = fig.add_subplot(gs[0])
    color_map_latent = plt.cm.Spectral(np.linspace(0, 1, len(np.unique(filtered_labels))))
    scatter = ax.scatter(latent_data_transformed[:, 0], latent_data_transformed[:, 1], s=30, c=filtered_labels, cmap='Spectral')

    gene_cell_types = ["neutrophil gene", "myeloid gene", "monocyte gene", "lung neutrophil gene",
                       "blood neutrophil gene"]
    gene_colors = ['green', 'orange', 'purple', 'brown', 'pink']
    gene_markers = ['^', 's', 'p', '*', 'D']

    for i, gene_type in enumerate(gene_cell_types):
        idxs = (gene_data_labels == gene_type)
        ax.scatter(combined_gene_data_transformed[idxs, 0], combined_gene_data_transformed[idxs, 1],
                   s=10, c=gene_colors[i], marker=gene_markers[i], label=gene_type, alpha=0.5)
    ax.set_aspect('equal')


    combined_x = np.concatenate([latent_data_transformed[:, 0], combined_gene_data_transformed[:, 0]])
    combined_y = np.concatenate([latent_data_transformed[:, 1], combined_gene_data_transformed[:, 1]])

    x_min, x_max = np.min(combined_x), np.max(combined_x)
    y_min, y_max = np.min(combined_y), np.max(combined_y)

    zoom_factor = 0.40
    padding_factor = 0.3

    x_range = (x_max - x_min) * zoom_factor
    y_range = (y_max - y_min) * zoom_factor

    center_x = (x_max + x_min) / 2
    center_y = (y_max + y_min) / 2

    new_x_min = center_x - (x_range * (1 + padding_factor))
    new_x_max = center_x + (x_range * (1 + padding_factor))
    new_y_min = center_y - (y_range * (1 + padding_factor))
    new_y_max = center_y + (y_range * (1 + padding_factor))

    ax.set_xlim(new_x_min, new_x_max)
    ax.set_ylim(new_y_min, new_y_max)

    ax.set_title(f'Latent Space Representation with Genes)', fontsize=18)
    ax.set_xlabel('UMAP Dimension 1', fontsize=16)
    ax.set_ylabel('UMAP Dimension 2', fontsize=16)

    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis('off')


    unique_filtered_labels = np.unique(filtered_labels)
    filtered_class_names = [inverse_label_map[label] for label in unique_filtered_labels if label in inverse_label_map]
    color_map = plt.cm.Spectral(np.linspace(0, 1, len(unique_filtered_labels)))
    latent_legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=filtered_class_names[i],
                                 markerfacecolor=color_map[i], markersize=18) for i in range(len(filtered_class_names))]

    gene_cell_types = ["neutrophil gene", "myeloid gene", "monocyte gene", "lung neutrophil gene",
                       "blood neutrophil gene"]

    gene_colors = ['green', 'orange', 'purple', 'brown', 'pink']  # Assuming 5 gene cell types
    gene_markers = ['^', 's', 'p', '*', 'D']  # Different marker styles for each gene cell type
    gene_legend_handles = [
        plt.Line2D([0], [0], marker=gene_markers[i], color='w', label=gene_cell_types[i],
                   markerfacecolor=gene_colors[i], markersize=10)
        for i in range(len(gene_cell_types))
    ]
    legend_handles = latent_legend_handles + gene_legend_handles

    ax_legend.legend(handles=legend_handles, loc='center', fontsize=12, title='Cell Types')

    plt.tight_layout()
    umap_figure_filename = os.path.join(umap_dir, f'umap_epoch_{epoch}.png')
    plt.savefig(umap_figure_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

"""
    inverse_label_map = {v: k for k, v in label_map.items()}  # inverse mapping for UMAP
    batch_size = 128
    num_classes = len(label_map)
    epoch = 140

    umap_dir = 'umap_mmd_job2'
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
    latent_dir = 'latent_data4cp2_new5_std'
    latents_path = os.path.join(latent_dir, f'latent_epoch_{epoch}.npy')
    label_dir = 'label_data4cp2_new5_std'
    labels_path = os.path.join(label_dir, f'label_epoch_141.npy')

    # Load all latent representations
    latent_data = np.load(latents_path)
    print(latent_data.shape)
    all_labels = np.load(labels_path)
    all_labels_array = np.array(all_labels)
    # print("Labels array shape:", all_labels_array.shape)

    # Filter out the 'erythroblast' class
    erythroblast_class_index = label_map['erythroblast']
    mask = all_labels_array != erythroblast_class_index
    filtered_latent_data = latent_data[mask]
    filtered_labels = all_labels_array[mask]

    # UMAP for latent space
    latent_data_umap = UMAP(n_neighbors=13, min_dist=0.1, n_components=2, metric='euclidean').fit_transform(
        filtered_latent_data)

    # UMAP for latent space
    latent_data_umap = UMAP(n_neighbors=13, min_dist=0.1, n_components=2, metric='euclidean').fit_transform(
        filtered_latent_data)

    plt.figure(figsize=(12, 10), dpi=150)
    scatter = plt.scatter(latent_data_umap[:, 0], latent_data_umap[:, 1], s=1, c=filtered_labels, cmap='Spectral')

    color_map = plt.cm.plasma(np.linspace(0, 1, len(set(filtered_labels))))
    class_names = [inverse_label_map[i] for i in range(len(inverse_label_map))]

    unique_filtered_labels = np.unique(filtered_labels)
    filtered_class_names = [inverse_label_map[label] for label in unique_filtered_labels if label in inverse_label_map]
    color_map = plt.cm.Spectral(np.linspace(0, 1, len(unique_filtered_labels)))

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=filtered_class_names[i],
                                 markerfacecolor=color_map[i], markersize=18) for i in range(len(filtered_class_names))]
    plt.legend(handles=legend_handles, loc='lower right', title='Cell Types')

    plt.title(f'Latent Space Representation - (Epoch {epoch})', fontsize=18)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)

    umap_figure_filename = os.path.join(umap_dir, f'umap_epoch_{epoch}_job2.png')

    # Save the UMAP figure
    plt.savefig(umap_figure_filename, dpi=300)
    print("it is saved")
    plt.close()

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
    
     latent_dir = '/Users/gizem/Downloads/latent_epoch_150 (2).npy'
    latents_path = '/Users/gizem/Downloads/latent_epoch_150 (2).npy'
    label_dir = '/Users/gizem/Downloads/label_epoch_151 (1).npy'
    labels_path = '/Users/gizem/Downloads/label_epoch_151 (1).npy'
    latent_data = np.load(latents_path)
    # latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
    print("Latent data shape:", latent_data.shape)
    all_labels_array = np.load(labels_path)
    print("Labels array shape:", all_labels_array.shape)


    # gene latents
    z_dir_ge = 'z_variables_GE_3'
    epoch_of_gen = 290

    class_labels_gen = [0, 1, 2, 3]
    # monocyte : class 1
    # myeloblast : class 2
    # basophil : class 0
    # neutrophil : class 3

    class_label_n = 3
    class_label_m = 1
    class_label_myeloid = 2

    z_filename_n = '/Users/gizem/Downloads/class_3_z_epoch_290.npy'
    z_filename_myeloid = '/Users/gizem/Downloads/class_2_z_epoch_290.npy'
    ref_z_class_neutrophils = np.load(z_filename_n)
    ref_z_class_myeloid = np.load(z_filename_myeloid)
    z_filename_m = '/Users/gizem/Downloads/class_1_z_epoch_290.npy'
    ref_z_class_mono = np.load(z_filename_m)

    latent_dir_ge_2 = 'latent_variables_GE_3_cp'
    z_dir_ge_2 = 'z_variables_GE_3_cp'
    class_label_n_blood = 2
    class_label_n_lung = 5

    z_filename_n_blood = '/Users/gizem/Downloads/class_2_z_epoch_290 (1).npy'
    ref_z_class_n_blood = np.load(z_filename_n_blood)
"""
