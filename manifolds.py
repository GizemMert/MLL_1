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
from Beta_Visualization import Beta as beta_p
from geomstats.information_geometry.beta import BetaDistributions as beta
from discrete_curve import DiscreteCurveViz
import numpy as np
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.discrete_curves import DiscreteCurves
from Stiefel_Manifold import StiefelSphere, StiefelCircle, Arrow2D
import matplotlib.pyplot as plt
from math import cos, sin
import numpy as np

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats.geometry.stiefel import Stiefel, StiefelCanonicalMetric
np.random.seed(seed=12)


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

beta_dir = 'beta_manifold_path'
if not os.path.exists(beta_dir):
    os.makedirs(beta_dir)

pdf_dir = 'pdf_manifold_path'
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)

discrete_dir = 'discrete_manifold_path'
if not os.path.exists(discrete_dir):
    os.makedirs(discrete_dir)

stiefel_dir = 'stiefel_manifold_path'
if not os.path.exists(stiefel_dir):
    os.makedirs(stiefel_dir)

grassman_dir = 'grassman_manifold_path'
if not os.path.exists(grassman_dir):
    os.makedirs(grassman_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoencodermodel4(latent_dim=30)
model_save_path = 'trained_model4cp2_new5.pth'
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()

train_dataset = Dataloader(split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=30, shuffle=False, num_workers=1)

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

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

# Prepare color coding for the geodesic
cc = gs.zeros((20, 3))
cc[:, 2] = gs.linspace(0, 1, 20)

# Plot the geodesic using the random points
beta_p.plot_geodesic(ax, random_myeloblast_point, random_neutrophil_banded_point, n_points=20, color=cc)

# Set plot limits and title
ax.set_xlim(0.0, 4.0)
ax.set_ylim(0.0, 4.0)
ax.set_title("Geodesic between two points for the Fisher-Rao metric")

beta_manifold_filename = os.path.join(beta_dir, f'beta_epoch_{epoch}.png')
plt.savefig(beta_manifold_filename, dpi=300)
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

