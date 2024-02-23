import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from PIL import Image
import os
from matplotlib.gridspec import GridSpec
from model4 import VariationalAutoencodermodel4, reparametrize
from Dataloader_2 import Dataloader
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import geomstats.backend as gs
import torch
import numpy as np
import cv2
import umap
import matplotlib.pyplot as plt
from geomstats.information_geometry.normal import NormalDistributions
import geomstats.geometry.complex_manifold as cm

# dimension = 30
# complex_manifold = cm.ComplexManifold(dimension)

normal = NormalDistributions(sample_dim=1)
epoch = 140
latent_dim = 30


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoencodermodel4(latent_dim=30)
model_save_path = 'trained_model4cp2_new5_std.pth'
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()

umap_dir = 'umap_path_3d'
if not os.path.exists(umap_dir):
    os.makedirs(umap_dir)

# Load all latent representations
latent_dir = 'latent_data4cp2_new5_std'
latents_path = os.path.join(latent_dir, f'latent_epoch_{epoch}.npy')
label_dir = 'label_data4cp2_new5_std'
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
filtered_latent_data = latent_data[mask]
print("filtered data shape:", filtered_latent_data.shape)
filtered_labels = all_labels_array[mask]

myeloblast_indices = np.where(filtered_labels == label_map['myeloblast'])[0]
neutrophil_banded_indices = np.where(filtered_labels == label_map['neutrophil_banded'])[0]

# np.random.seed(42)
random_myeloblast_index = np.random.choice(myeloblast_indices)
random_neutrophil_banded_index = np.random.choice(neutrophil_banded_indices)

random_myeloblast_point = filtered_latent_data[random_myeloblast_index]
random_neutrophil_banded_point = filtered_latent_data[random_neutrophil_banded_index]
print("Poin data shape:", random_myeloblast_point.shape)

def interpolate_gpr(latent_start, latent_end, n_points=100):
    if isinstance(latent_start, torch.Tensor):
        latent_start = latent_start.detach().cpu().numpy()
    if isinstance(latent_end, torch.Tensor):
        latent_end = latent_end.detach().cpu().numpy()

    indices = np.array([0, 1]).reshape(-1, 1)


    latent_vectors = np.vstack([latent_start, latent_end])

    kernel = C(1.0, (1e-1, 1e1)) * RBF(1e-1, (1e-1, 1e1))

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(indices, latent_vectors)

    index_range = np.linspace(0, 1, n_points).reshape(-1, 1)

    interpolated_latent_vectors = gpr.predict(index_range)

    return interpolated_latent_vectors

unique_labels = np.unique(filtered_labels)

interpolated_latents = interpolate_gpr(random_myeloblast_point, random_neutrophil_banded_point, n_points=100)
# Perform UMAP dimensionality reduction to 3D
reducer = umap.UMAP(n_components=3)
latent_data_3d = reducer.fit_transform(filtered_latent_data)

interpolated_latents_3d = reducer.transform(interpolated_latents)

# Create a Plotly interactive 3D scatter plot for the original data
trace_data = go.Scatter3d(
    x=latent_data_3d[:, 0],
    y=latent_data_3d[:, 1],
    z=latent_data_3d[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=filtered_labels,  # Color by labels
        colorscale='Spectral',  # Choose a colorscale
        opacity=0.8
    ),
    name='Data Points'
)

# Create a trace for the interpolation path
trace_path = go.Scatter3d(
    x=interpolated_latents_3d[:, 0],
    y=interpolated_latents_3d[:, 1],
    z=interpolated_latents_3d[:, 2],
    mode='lines',
    line=dict(
        color='white',  # Line color can be adjusted
        width=10
    ),
    name='Interpolation Path'
)

# Combine the data and the path for plotting
data = [trace_data, trace_path]

# Define the layout of the plot
layout = go.Layout(
    title="3D UMAP Visualization",
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        xaxis=dict(title='UMAP Dimension 1'),
        yaxis=dict(title='UMAP Dimension 2'),
        zaxis=dict(title='UMAP Dimension 3'),
    ),
    legend=dict(itemsizing='constant')  # Ensure constant legend item size
)

# Create the figure with data and layout
fig = go.Figure(data=data, layout=layout)
fig.show()

# Save the figure to an HTML file
fig.write_html("3d_umap_interpolation_gen.html")

