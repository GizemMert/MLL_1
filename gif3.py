from PIL import Image
import torch
import numpy as np
from model4 import VariationalAutoencodermodel4, reparametrize
from Dataloader_2 import Dataloader
from torch.utils.data import DataLoader
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
from sklearn.gaussian_process.kernels import WhiteKernel
import os

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
epoch = 140
latent_dim = 30
inverse_label_map = {v: k for k, v in label_map.items()}  # inverse mapping for UMAP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoencodermodel4(latent_dim=30)
model_save_path = 'trained_model4cp2_new5.pth'
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()

umap_dir = 'umap_path'
if not os.path.exists(umap_dir):
    os.makedirs(umap_dir)

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


def interpolate_gif_with_gpr(model, filename, latent_points, n=100, latent_dim=30):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    def get_latent_vector(x):
        distributions = model.encoder(x)
        mu = distributions[:, :latent_dim]
        logvar = distributions[:, latent_dim:]
        z = reparametrize(mu, logvar)
        return z

    # Generate the latent representations
    latents = [get_latent_vector(feature.to(device)) for feature in features]
    """

    def slerp(val, low, high):
        low_norm = low / torch.norm(low)
        high_norm = high / torch.norm(high)
        omega = torch.acos((low_norm * high_norm).sum().clamp(-1, 1))
        so = torch.sin(omega)
        res = torch.sin((1.0 - val) * omega) / so * low + torch.sin(val * omega) / so * high
        return res.where(so != 0, low)

    # Interpolate between the latent vectors
    all_interpolations = []
    for i in range(len(latent_points) - 1):
        for t in np.linspace(0, 1, n):
            z_interp = slerp(t, latent_points[i], latent_points[i + 1])
            all_interpolations.append(z_interp)


    interpolate_list = []
    for z in all_interpolations:
        y = model.decoder(z)
        img = model.img_decoder(y)
        img = img.permute(0, 2, 3, 1)
        interpolate_list.append(img.squeeze(0).to('cpu').detach().numpy())


    interpolate_list = [np.clip(img * 255, 0, 255).astype(np.uint8) for img in interpolate_list]

    images_list = [Image.fromarray(img) for img in interpolate_list]


    images_list[0].save(
        f'{filename}.gif',
        save_all=True,
        append_images=images_list[1:],
        loop=0,
        duration=100
    )

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoencodermodel4(latent_dim=30)
model_save_path = 'trained_model4cp2_new5.pth'
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()


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

# Convert to appropriate format and device
selected_images = [feature.float().to(device) for feature in selected_features if feature is not None]

latent_points = [
    torch.from_numpy(random_myeloblast_point).float().to(device),
    torch.from_numpy(random_neutrophil_banded_point).float().to(device)]
interpolate_gif_with_gpr(model, "vae_interpolation_latent_space", latent_points)
