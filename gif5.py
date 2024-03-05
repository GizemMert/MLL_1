import os
import pickle
import imageio
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from PIL import Image
from model4 import VariationalAutoencodermodel4, reparametrize
from Dataloader_4 import Dataloader
from torch.utils.data import DataLoader
import cv2
import umap
import matplotlib.pyplot as plt
from Model_Vae_GE_2 import VAE_GE
from umap import UMAP
import seaborn as sns
import pandas as pd
import anndata
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.neighbors import NearestNeighbors
import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import numpy as np

# dimension = 30
# complex_manifold = cm.ComplexManifold(dimension)


epoch = 280
latent_dim = 50

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_1 = VariationalAutoencodermodel4(latent_dim=50)
model_save_path = 'trained_model4cp2_new5_std_gen_2.pth'
model_1.load_state_dict(torch.load(model_save_path, map_location=device))
model_1.to(device)
model_1.eval()

model_2 = VAE_GE(input_shape=2432, latent_dim=50).to(device)
model_save_path_2 = 'trained_model_GE_3_cp.pth'
model_2.load_state_dict(torch.load(model_save_path_2, map_location=device))
model_2.to(device)
model_2.eval()

umap_dir = 'umap_trajectory_lung_blood'
if not os.path.exists(umap_dir):
    os.makedirs(umap_dir)

class_label_n_blood = 2
class_label_n_liver = 1
class_label_n_lung = 5
class_label_m = 0
class_label_myeloid = 4

# Load all latent representations
latent_dir = 'latent_data4cp2_new5_std_gen_2_cp'
latents_path = os.path.join(latent_dir, f'latent_epoch_{epoch}.npy')
label_dir = 'label_data4cp2_new5_std_gen_2_cp'
labels_path = os.path.join(label_dir, f'label_epoch_281.npy')
n_banded_lung_z_dir = 'banded_z_data4cp2_new5_std_gen_2_cp'
n_banded_lung_z_path = os.path.join(n_banded_lung_z_dir, f'neutrophil_z_band_{epoch}.npy')
n_segment_blood_z_dir = 'segment_z_data4cp2_new5_std_gen_2_cp'
n_segment_blood_lung_z_path = os.path.join(n_segment_blood_z_dir, f'neutrophil_z_segment_{epoch}.npy')

# Load all latent representations
latent_data = np.load(latents_path)
# latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
print("Latent data shape:", latent_data.shape)

n_banded_data = np.load(n_banded_lung_z_path)
# latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
print("Neutrophil Banded Latent data shape:", n_banded_data.shape)

n_segment_data = np.load(n_segment_blood_lung_z_path)
# latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
print("N.Segmented Latent data shape:", n_segment_data.shape)

# Load all labels
all_labels_array = np.load(labels_path)
print("Labels array shape:", all_labels_array.shape)

# print("Labels array shape:", all_labels_array.shape)

# Filter out the 'erythroblast' class
erythroblast_class_index = label_map['erythroblast']
neutrophil_banded_index = label_map['neutrophil_banded']
segmented_index = label_map['neutrophil_segmented']
mask = all_labels_array != erythroblast_class_index
mask2 = (all_labels_array == neutrophil_banded_index)
mask3 = (all_labels_array == segmented_index)
filtered_latent_data = latent_data[mask]
print("filtered data shape:", filtered_latent_data.shape)
filtered_labels = all_labels_array[mask]
filtered_labels_neutrophil_banded = all_labels_array[mask2]
filtered_labels_neutrophil_segmented = all_labels_array[mask3]

print("filtered neutrophil label shape:", filtered_labels_neutrophil_banded.shape)
print("filtered neutrophil label shape:", filtered_labels_neutrophil_segmented.shape)

myeloblast_indices = np.where(filtered_labels == label_map['myeloblast'])[0]
neutrophil_banded_indices = np.where(filtered_labels_neutrophil_banded == label_map['neutrophil_banded'])[0]
neutrophil_seg_indices = np.where(filtered_labels_neutrophil_segmented == label_map['neutrophil_segmented'])[0]
basophil_indices = np.where(filtered_labels == label_map['basophil'])[0]
eosinophil_indices = np.where(filtered_labels == label_map['eosinophil'])[0]
monocyte_indices = np.where(filtered_labels == label_map['monocyte'])[0]

# np.random.seed(10)
random_myeloblast_index = np.random.choice(myeloblast_indices)
random_neutrophil_banded_index = np.random.choice(neutrophil_banded_indices)
random_neutrophil_seg_index = np.random.choice(neutrophil_seg_indices)
random_basophil_index = np.random.choice(basophil_indices)
random_eosinophil_index = np.random.choice(eosinophil_indices)
random_monocyte_index = np.random.choice(monocyte_indices)

random_myeloblast_point = filtered_latent_data[random_myeloblast_index]
# You can replace filtered_laten_data with neutrophil_data
random_neutrophil_banded_point = n_banded_data[random_neutrophil_banded_index]
random_neutrophil_seg_point = n_segment_data[random_neutrophil_seg_index]
random_basophil_point = filtered_latent_data[random_basophil_index]
random_eosinophil_point = filtered_latent_data[random_eosinophil_index]
random_monocyte_point = filtered_latent_data[random_monocyte_index]


# print("Point data shape:", random_myeloblast_point.shape)


def interpolate_gpr(latent_start, latent_end, steps=100):
    if isinstance(latent_start, torch.Tensor):
        latent_start = latent_start.detach().cpu().numpy()
    if isinstance(latent_end, torch.Tensor):
        latent_end = latent_end.detach().cpu().numpy()

    indices = np.array([0, 1]).reshape(-1, 1)

    latent_vectors = np.vstack([latent_start, latent_end])

    kernel = C(1.0, (1e-1, 1e1)) * RBF(1e-1, (1e-1, 1e1))

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(indices, latent_vectors)

    index_range = np.linspace(0, 1, steps).reshape(-1, 1)

    interpolated_latent_vectors = gpr.predict(index_range)

    return interpolated_latent_vectors


def interpolate_gif_gpr(model, filename, latent_start, latent_end, steps=100, grid_size=(10, 10), device=device):
    model_1.eval()

    interpolated_latent_points = interpolate_gpr(latent_start, latent_end, steps=steps)

    file_path = 'interpolation_band_segmented'
    torch.save(interpolated_latent_points, file_path + '_latent_points.pt')
    print(f"Interpolated latent points saved to {file_path}_latent_points.pt")

    decoded_images = []
    for z in interpolated_latent_points:
        z_tensor = torch.from_numpy(z).float().to(device).unsqueeze(0)
        with torch.no_grad():
            decoded_img = model_1.decoder(z_tensor)
            decoded_img = model_1.img_decoder(decoded_img)
        decoded_images.append(decoded_img.cpu())

    while len(decoded_images) < grid_size[0] * grid_size[1]:
        decoded_images.append(torch.zeros_like(decoded_images[0]))
    decoded_images = decoded_images[:grid_size[0] * grid_size[1]]

    tensor_grid = torch.stack(decoded_images).squeeze(1)  # Remove batch dimension if necessary
    grid_image = make_grid(tensor_grid, nrow=grid_size[1], normalize=True, padding=2)
    grid_image = ToPILImage()(grid_image)
    grid_image.save(filename + '.jpg', quality=400)
    print("Grid Image saved successfully")


def get_images_from_different_classes(dataloader, class_1_label, class_2_label):
    feature_1, feature_2 = None, None

    for feature, _, _, labels, _ in dataloader:
        if feature_1 is not None and feature_2 is not None:
            break

        for i, label in enumerate(labels):
            if label.item() == class_1_label and feature_1 is None:
                feature_1 = feature[i].unsqueeze(0)

            if label.item() == class_2_label and feature_2 is None:
                feature_2 = feature[i].unsqueeze(0)

    return [feature_1, feature_2]


def get_latent_vector(x):
    distributions = model_1.encoder(x)
    print(f"Distributions shape: {distributions.shape}")
    mu = distributions[:, :50]
    logvar = distributions[:, 50:100]
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    z = reparametrize(mu, logvar)
    print("Shape of z:", z.shape)
    return z


train_dataset = Dataloader(split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)

selected_features = get_images_from_different_classes(train_dataloader, label_map['neutrophil_banded'],
                                                      label_map['neutrophil_segmented'])

start_latent, end_latent = [get_latent_vector(feature.float().to(device)) for feature in selected_features]
# interpolate_gif_gpr("interpolation_img_ge", start_latent, end_latent, steps=100, grid_size=(10, 10), device=device)
interpolate_gif_gpr(model_1, "grid_banded_segmented", random_neutrophil_banded_point, random_neutrophil_seg_point,
                    steps=100, grid_size=(10, 10))
