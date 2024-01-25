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
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.geometry.connection import Connection
import geomstats.visualization as visualization
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.information_geometry.normal import NormalDistributions
from geomstats.geometry.hyperboloid import Hyperboloid
import geomstats.geometry.hyperbolic
from matplotlib.lines import Line2D

hyperbolic = Hyperboloid(dim=2)


if __name__ == '__main__':
    normal = NormalDistributions(sample_dim=1)


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

    latent_data_umap = UMAP(n_neighbors=13, min_dist=0.1, n_components=2, metric='euclidean').fit_transform(
        filtered_latent_data)

    myeloblast_umap_points = latent_data_umap[filtered_labels == label_map['myeloblast']]
    neutrophil_banded_umap_points = latent_data_umap[filtered_labels == label_map['neutrophil_banded']]

    random_myeloblast_point = myeloblast_umap_points[np.random.choice(myeloblast_umap_points.shape[0])]
    random_neutrophil_banded_point = neutrophil_banded_umap_points[
        np.random.choice(neutrophil_banded_umap_points.shape[0])]

    initial_point = hyperbolic.projection(gs.array([0.0, random_myeloblast_point[0], random_myeloblast_point[1]]))
    end_point = hyperbolic.projection(
        gs.array([0.0, random_neutrophil_banded_point[0], random_neutrophil_banded_point[1]]))

    assert hyperbolic.belongs(initial_point).all(), "Initial point not on the hyperboloid."
    assert hyperbolic.belongs(end_point).all(), "End point not on the hyperboloid."

    geodesic_func = hyperbolic.metric.geodesic(
        initial_point=initial_point, end_point=end_point
    )

    points = geodesic_func(gs.linspace(0.0, 1.0, 10))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    representation = "S2"

    ax = visualization.plot(
        initial_point, ax=ax, space=representation, s=50, label="Initial point", color='blue'
    )
    ax = visualization.plot(end_point, ax=ax, space=representation, s=50, label="End point", color='orange')

    ax = visualization.plot(
        points[1:-1], ax=ax, space=representation, s=5, color="black", label="Geodesic"
    )
    ax.set_title("Geodesic on the hyperbolic plane in Poincare disk representation")

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Myeloblast (Initial Point)',
                   markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Neutrophil Banded (End Point)',
                   markerfacecolor='orange', markersize=10),
    ]

    # Add the legend to the plot using the handles
    plt.legend(handles=legend_handles, loc='lower right')

    pdf_figure_filename = os.path.join(beta_dir, f'Sphere_interpolation_epoch_{epoch}.png')
    plt.savefig(pdf_figure_filename)
    plt.close(fig)