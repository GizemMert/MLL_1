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
import os
import umap
import matplotlib.pyplot as plt
from geomstats.information_geometry.normal import NormalDistributions
import geomstats.geometry.complex_manifold as cm

dimension = 30
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


"""
# Load all latent representations
latent_dir = 'latent_data4cp2_new5'
latents_path = '/Users/gizem/MLL_1/latent_epoch_140 (1).npy'
label_dir = 'label_data4cp2_new5'
labels_path = '/Users/gizem/MLL_1/label_epoch_140 (1).npy'

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

# UMAP for latent space
# latent_data_umap = umap.UMAP(n_neighbors=13, min_dist=0.1, n_components=2, metric='euclidean').fit_transform(filtered_latent_data)

myeloblast_indices = np.where(filtered_labels == label_map['myeloblast'])[0]
neutrophil_banded_indices = np.where(filtered_labels == label_map['neutrophil_banded'])[0]

# np.random.seed(42)
# Select random latent vectors for myeloblast and neutrophil banded points
random_myeloblast_index = np.random.choice(myeloblast_indices)
random_neutrophil_banded_index = np.random.choice(neutrophil_banded_indices)

random_myeloblast_point = filtered_latent_data[random_myeloblast_index]
random_neutrophil_banded_point = filtered_latent_data[random_neutrophil_banded_index]
print("Poin data shape:", random_myeloblast_point.shape)
"""


def get_latent_vector(x):
    distributions = model.encoder(x)
    mu = distributions[:, :latent_dim]
    logvar = distributions[:, latent_dim:]
    z = reparametrize(mu, logvar)
    return z


"""
def to_complex_coordinates(latent_vector):
    real_part = latent_vector[:latent_vector.shape[0] // 2]
    imag_part = latent_vector[latent_vector.shape[0] // 2:]
    return real_part + 1j * imag_part

"""
def interpolate_gpr(latent_start, latent_end, n_points=20):
    if isinstance(latent_start, torch.Tensor):
        latent_start = latent_start.detach().cpu().numpy()
    if isinstance(latent_end, torch.Tensor):
        latent_end = latent_end.detach().cpu().numpy()

    indices = np.array([0, 1]).reshape(-1, 1)


    latent_vectors = np.vstack([latent_start, latent_end])

    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e2))

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

selected_features = get_images_from_different_classes(train_dataloader, label_map['myeloblast'], label_map['neutrophil_banded'])

start_latent, end_latent = [get_latent_vector(feature.float().to(device),) for feature in selected_features]

# Call the function with your data
interpolate_gif_gpr("vae_interpolation_gpr", start_latent, end_latent, steps=100, grid_size=(10, 10))

""""
def get_latent_vector(x):
    distributions = model.encoder(x)
    mu = distributions[:, :latent_dim]
    logvar = distributions[:, latent_dim:]
    z = reparametrize(mu, logvar)
    return z
    
def compute_geodesic_path(start_latent, end_latent, n_points=20):
    print(f"Start latent shape: {start_latent.shape}")
    print(f"End latent shape: {end_latent.shape}")

    geodesic_ab_fisher = normal.metric.geodesic(start_latent, end_latent)
    t = gs.linspace(0, 1, n_points)
    geodesic_path = geodesic_ab_fisher(t)

    print(f"Geodesic path shape: {geodesic_path.shape}")

def interpolate_gif_pdf(filename, start_latent, end_latent, steps=20, grid_size=(30, 10)):
    print("Starting interpolate_gif_pdf function")
    model.eval()

    geodesic_path = compute_geodesic_path(start_latent.squeeze(), end_latent.squeeze(), steps)
    print("Computed geodesic path")

    decoded_images = []
    for i, z in enumerate(geodesic_path):
        print(f"Processing step {i+1}/{len(geodesic_path)}")

        # Convert numpy array to torch tensor and ensure correct shape
        z_tensor = torch.from_numpy(z).float().to(device).unsqueeze(0)

        # Print the shape of z_tensor to debug
        print(f"Shape of z_tensor: {z_tensor.shape}")
        with torch.no_grad():
            decoded_img = model.decoder(z_tensor)
            decoded_img = model.img_decoder(decoded_img)
        decoded_images.append(decoded_img.cpu())

    print(f"Total decoded images: {len(decoded_images)}")
    while len(decoded_images) < grid_size[0] * grid_size[1]:
        decoded_images.append(torch.zeros_like(decoded_images[0]))

    decoded_images = decoded_images[:grid_size[0] * grid_size[1]]

    tensor_grid = torch.stack(decoded_images).squeeze(1)  # Remove batch dimension if necessary
    grid_image = make_grid(tensor_grid, nrow=grid_size[1], normalize=True, padding=2)
    grid_image = ToPILImage()(grid_image)
    grid_image.save('/path/to/directory/' + filename + '.jpg', quality=95)
    print("Image saved successfully")


interpolate_gif_pdf("vae_interpolation_pdf", random_myeloblast_point, random_neutrophil_banded_point, steps=20, grid_size=(30, 10))


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


