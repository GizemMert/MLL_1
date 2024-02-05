import os

from scipy.spatial import KDTree

os.environ['PYDEVD_USE_CYTHON'] = 'NO'
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from PIL import Image
import torch
import numpy as np
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoencodermodel4(latent_dim=30)
model_save_path = 'trained_model4cp2_new5.pth'
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()


# Load all latent representations
latent_dir = 'latent_data4cp2_new5'
latents_path = os.path.join(latent_dir, f'latent_epoch_{epoch}.npy')
label_dir = 'label_data4cp2_new5'
labels_path = os.path.join(label_dir, f'label_epoch_{epoch}.npy')

# Load all latent representations
latent_data = np.load(latents_path)
latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
print("Latent data shape:", latent_data_reshaped.shape)

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
neutrophil_seg_indices = np.where(filtered_labels == label_map['neutrophil_segmented'])[0]
basophil_indices = np.where(filtered_labels == label_map['basophil'])[0]
eosinophil_indices = np.where(filtered_labels == label_map['eosinophil'])[0]
monocyte_indices = np.where(filtered_labels == label_map['monocyte'])[0]

np.random.seed(10)
random_myeloblast_index = np.random.choice(myeloblast_indices)
random_neutrophil_banded_index = np.random.choice(neutrophil_banded_indices)
random_neutrophil_seg_index = np.random.choice(neutrophil_seg_indices)
random_basophil_index = np.random.choice(basophil_indices)
random_eosinophil_index = np.random.choice(eosinophil_indices)
random_monocyte_index = np.random.choice(monocyte_indices)

random_myeloblast_point = filtered_latent_data[random_myeloblast_index]
random_neutrophil_banded_point = filtered_latent_data[random_neutrophil_banded_index]
random_neutrophil_seg_point = filtered_latent_data[random_neutrophil_seg_index]
random_basophil_point = filtered_latent_data[random_basophil_index]
random_eosinophil_point = filtered_latent_data[random_eosinophil_index]
random_monocyte_point = filtered_latent_data[random_monocyte_index]
print("Poin data shape:", random_myeloblast_point.shape)


def interpolate_gpr(latent_start, latent_end, n_points=100):
    if isinstance(latent_start, torch.Tensor):
        latent_start = latent_start.detach().cpu().numpy()
    if isinstance(latent_end, torch.Tensor):
        latent_end = latent_end.detach().cpu().numpy()

    indices = np.array([0, 1]).reshape(-1, 1)


    latent_vectors = np.vstack([latent_start, latent_end])

    kernel = C(1.0, (1e-1, 1e1)) * RBF(1e-1, (1e-1, 1e1))

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
    gpr.fit(indices, latent_vectors)

    index_range = np.linspace(0, 1, n_points).reshape(-1, 1)

    interpolated_latent_vectors = gpr.predict(index_range)

    return interpolated_latent_vectors




def interpolate_gif_gpr(filename, start_latent, end_latent, steps=100, grid_size=(10, 10),
                        device='cpu'):


    model.eval()
    # Compute interpolated latent vectors using GPR
    interpolated_latents = interpolate_gpr(start_latent, end_latent, steps)

    decoded_images = []
    for z in interpolated_latents:
        z_tensor = torch.from_numpy(z).float().to(device).unsqueeze(0)
        with torch.no_grad():
            decoded_img = model.decoder(z_tensor)
            decoded_img = model.img_decoder(decoded_img)
        decoded_images.append(decoded_img.cpu())

    # Ensure the decoded images fit into the specified grid size
    while len(decoded_images) < grid_size[0] * grid_size[1]:
        decoded_images.append(torch.zeros_like(decoded_images[0]))
    decoded_images = decoded_images[:grid_size[0] * grid_size[1]]

    # Arrange images in a grid and save
    tensor_grid = torch.stack(decoded_images).squeeze(1)  # Remove batch dimension if necessary
    grid_image = make_grid(tensor_grid, nrow=grid_size[1], normalize=True, padding=2)
    grid_image = ToPILImage()(grid_image)
    grid_image.save(filename + '.jpg', quality=95)
    print("Image saved successfully")



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
    distributions = model.encoder(x)
    mu = distributions[:, :latent_dim]
    logvar = distributions[:, latent_dim:]
    z = reparametrize(mu, logvar)
    return z


train_dataset = Dataloader(split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)

selected_features = get_images_from_different_classes(train_dataloader, label_map['myeloblast'], label_map['neutrophil_banded'])

start_latent, end_latent = [get_latent_vector(feature.float().to(device)) for feature in selected_features]

interpolate_gif_gpr("vae_interpolation_gpr_BAS", random_myeloblast_point, random_eosinophil_point, steps=100, grid_size=(10, 10))




"""
def get_latent_vector(x):
    distributions = model.encoder(x)
    mu = distributions[:, :latent_dim]
    logvar = distributions[:, latent_dim:]
    z = reparametrize(mu, logvar)
    return z


def compute_class_centroid(dataloader, class_label, model, device, latent_dim):
    latent_vectors = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            # Unpack the batch
            feat, _, _, labels, _ = batch
            feat = feat.to(device)
            labels = labels.to(device)

            # Filter the data for the desired class
            class_data = feat[labels == class_label]

            if len(class_data) > 0:
                distributions = model.encoder(class_data)
                mu = distributions[:, :latent_dim]
                latent_vectors.append(mu.cpu().numpy())

    if latent_vectors:
        centroid = np.mean(np.vstack(latent_vectors), axis=0)
    else:
        centroid = np.zeros(latent_dim)
    return centroid


def interpolate_centroids(centroid1, centroid2, steps=100):
    interpolated_latents = np.linspace(centroid1, centroid2, num=steps)
    return interpolated_latents







def interpolate_gpr(latent_start, latent_end, n_points=20):
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

def interpolate_gif_gpr(filename, start_latent, end_latent, steps=100, grid_size=(10, 10)):
    model.eval()

    # Compute interpolated latent vectors using GPR
    interpolated_latents = interpolate_gpr(start_latent, end_latent, steps)

    decoded_images = []
    for i, z in enumerate(interpolated_latents):
        z_tensor = torch.from_numpy(z).float().to(device).unsqueeze(0)
        with torch.no_grad():
            decoded_img = model.decoder(z_tensor)
            decoded_img = model.img_decoder(decoded_img)
        decoded_images.append(decoded_img.cpu())

    total_slots = grid_size[0] * grid_size[1]
    while len(decoded_images) < total_slots:
        decoded_images.append(torch.zeros_like(decoded_images[0]))

    # Trim the list to match the grid size exactly
    decoded_images = decoded_images[:total_slots]

    # Arrange images in a grid
    tensor_grid = torch.stack(decoded_images).squeeze(1)  # Remove batch dimension if necessary
    grid_image = make_grid(tensor_grid, nrow=grid_size[1], normalize=True, padding=2)
    grid_image = ToPILImage()(grid_image)
    grid_image.save(filename + '.jpg', quality=95)
    print("Image saved successfully")



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


train_dataset = Dataloader(split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)

label_map = {'myeloblast': 0, 'neutrophil_banded': 1}  # Example label_map

centroid_class_1 = compute_class_centroid(train_dataloader, label_map['myeloblast'], model, device, 30)
centroid_class_2 = compute_class_centroid(train_dataloader, label_map['neutrophil_banded'], model, device, 30)
def interpolate_grid_com(filename, centroid1, centroid2, grid_size=(10, 10)):
    model.eval()

    # Compute interpolated latent vectors using GPR
    interpolated_latents = interpolate_centroids(centroid1, centroid2, steps=100)
    print("Number of interpolated latent vectors:", len(interpolated_latents))

    decoded_images = []
    for i, z in enumerate(interpolated_latents):
        z_tensor = torch.from_numpy(z).float().to(device).unsqueeze(0)
        with torch.no_grad():
            decoded_img = model.decoder(z_tensor)
            decoded_img = model.img_decoder(decoded_img)
        decoded_images.append(decoded_img.cpu())
        print(f"Decoded image {i + 1}/{len(interpolated_latents)}")

    print("Loop completed")

    if not decoded_images:
        print("No images to create a grid.")
        return
    print("Checked if decoded_images is empty")

    total_slots = grid_size[0] * grid_size[1]

    # Check if decoded_images is empty
    if not decoded_images:
        print("No images to create a grid.")
        return

    # Check if the number of decoded images is less than total_slots
    while len(decoded_images) < total_slots:
        decoded_images.append(torch.zeros_like(decoded_images[0]))

    # Trim the list to match the grid size exactly
    decoded_images = decoded_images[:total_slots]
    print("Trimmed the list of decoded images")

    # Arrange images in a grid
    print("Before while loop")
    tensor_grid = torch.stack(decoded_images, dim=0)
    tensor_grid = tensor_grid.squeeze(1)
    print("Updated tensor_grid shape:", tensor_grid.shape)
    print("Initial tensor_grid shape:", tensor_grid.shape)
    while tensor_grid.dim() > 4:
        tensor_grid = tensor_grid.squeeze(0)
        print("Current tensor_grid shape:", tensor_grid.shape)
    # tensor_grid = tensor_grid.permute(0, 2, 3, 1)

    # Make sure grid_size does not exceed the number of images
    # grid_size = (min(grid_size[0], len(decoded_images)), min(grid_size[1], len(decoded_images) // grid_size[0]))
    total_images = 100
    images_per_grid = grid_size[0] * grid_size[1]

    # Check if the total_images is equal to the grid size
    if total_images != images_per_grid:
        print(f"Warning: Total images ({total_images}) should match grid size ({images_per_grid}).")

    grid_image = make_grid(tensor_grid, nrow=grid_size[1], normalize=True, padding=0)
    grid_image = ToPILImage()(grid_image)
    grid_image.save(filename + '.jpg')
    print("Image saved successfully")

selected_features = get_images_from_different_classes(train_dataloader, label_map['myeloblast'], label_map['neutrophil_banded'])

start_latent, end_latent = [get_latent_vector(feature.float().to(device),) for feature in selected_features]

# Call the function with your data
interpolate_grid_com("vae_interpolation_com", centroid_class_1, centroid_class_2, grid_size=(10, 10))


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


train_dataset = Dataloader(split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)

selected_features = get_images_from_different_classes(train_dataloader, label_map['myeloblast'], label_map['neutrophil_banded'])

start_latent, end_latent = [get_latent_vector(feature.float().to(device),) for feature in selected_features]
"""


