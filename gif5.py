import os

from scipy.spatial import KDTree
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

gif_dir = 'gif_path'
if not os.path.exists(gif_dir):
    os.makedirs(gif_dir)

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


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def nearest_neighbor_interpolation_with_exemplars(start_point, end_point, latent_points, n_steps=100):

    tree = KDTree(latent_points)


    linear_steps = np.linspace(start_point, end_point, n_steps)


    path = [start_point]

    for step in linear_steps[1:-1]:
        _, nearest_index = tree.query(step)
        nearest_point = latent_points[nearest_index]


        path.append(nearest_point)


    path.append(end_point)

    return np.array(path)


def generate_image_grid(filename, latent_start, latent_end, latent_dataset, model, device, n_steps=100,
                        grid_size=(10, 10)):
    model.eval()


    interpolated_latents = nearest_neighbor_interpolation_with_exemplars(latent_start, latent_end, latent_dataset,
                                                                         n_steps)


    latent_tensors = torch.tensor(interpolated_latents).float().to(device)


    with torch.no_grad():

        decoded_images = model.decoder(latent_tensors).cpu()
        decoded_images = model.img_decoder(decoded_images) if hasattr(model, 'img_decoder') else decoded_images


    if len(decoded_images) < grid_size[0] * grid_size[1]:
        additional_images = torch.zeros((grid_size[0] * grid_size[1] - len(decoded_images), *decoded_images.shape[1:]),
                                        device='cpu')
        decoded_images = torch.cat([decoded_images, additional_images], dim=0)


    decoded_images = decoded_images[:grid_size[0] * grid_size[1]]

    # Arrange images in a grid
    grid_image = make_grid(decoded_images, nrow=grid_size[1], normalize=True, padding=2)
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

generate_image_grid("vae_interpolation_KNN", start_latent, end_latent, latent_dataset=filtered_latent_data,  steps=100, grid_size=(10, 10))


