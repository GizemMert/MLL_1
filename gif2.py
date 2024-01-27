from torch.utils.data import DataLoader
import torch
import umap
from Dataloader_4 import Dataloader, label_map
from model4 import VariationalAutoencodermodel4, reparametrize
import os
import plotly.graph_objs as go
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import geomstats.backend as gs
import matplotlib.pyplot as plt
from Beta_Visualization import Beta
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.geometry.connection import Connection
from geomstats.geometry.complex_manifold import ComplexManifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean

import matplotlib.pyplot as plt
import matplotlib


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
    # model.load_state_dict(torch.load(model_save_path, map_location=device))
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
    unique_labels = np.unique(filtered_labels)

    # Perform UMAP dimensionality reduction to 3D
    reducer = umap.UMAP(n_components=3)
    latent_data_3d = reducer.fit_transform(filtered_latent_data)

    # Create a Plotly interactive 3D scatter plot
    trace = go.Scatter3d(
        x=latent_data_3d[:, 0],
        y=latent_data_3d[:, 1],
        z=latent_data_3d[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=filtered_labels,  # Color by labels
            colorscale='Spectral',  # Choose a colorscale
            opacity=0.8
        )
    )

    data = [trace]
    layout = go.Layout(
        title="3D UMAP Visualization",
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title='UMAP Dimension 1'),
            yaxis=dict(title='UMAP Dimension 2'),
            zaxis=dict(title='UMAP Dimension 3'),
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()
    fig.write_html("your_plot.html")
