from PIL import Image
import torch
import numpy as np
from model4 import VariationalAutoencodermodel4, reparametrize
from Dataloader_2 import Dataloader
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
import torch
import numpy as np
from PIL import Image
from sklearn.gaussian_process.kernels import WhiteKernel

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
model_save_path = 'trained_model4cp2_new3.pth'
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()


def get_latent_vector(x, latent_dim=30):
    distributions = model.encoder(x)
    mu = distributions[:, :latent_dim]
    logvar = distributions[:, latent_dim:]
    z = reparametrize(mu, logvar)
    return z


def interpolate_gif(filename, start_latent, end_latent, latent_dim=30, steps_per_dim=10, grid_size=(30, 10)):
    model.eval()

    def interpolate_single_dimension(start, end, dim, steps=10):
        start_val = start[dim].item()  # Get the scalar value
        end_val = end[dim].item()  # Get the scalar value
        interp_values = torch.linspace(start_val, end_val, steps=steps, device=device)

        # Create a tensor to hold all interpolated points
        interpolated_dim = torch.stack(
            [start.clone().detach().requires_grad_(False).scatter_(0, torch.tensor([dim], device=device), interp_value)
             for interp_value in interp_values])

        return interpolated_dim

    interpolated_images = []
    for dim in range(latent_dim):
        # Interpolate each dimension separately and extend the list of images
        dim_interpolations = interpolate_single_dimension(start_latent.squeeze(), end_latent.squeeze(), dim, steps_per_dim)
        interpolated_images.extend(dim_interpolations.unbind(0))  # Unbind the 0-th dimension to get a list of images

    interpolated_images = interpolated_images[:grid_size[0] * grid_size[1]]

    decoded_images = []
    for z in interpolated_images:
        z = z.to(device).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            decoded_img = model.decoder(z)
            decoded_img = model.img_decoder(decoded_img)# This should output the reconstructed image
        decoded_images.append(decoded_img)

    tensor_grid = torch.stack(decoded_images).squeeze(1)  # Remove batch dimension
    # Normalize and convert the grid to a PIL Image
    grid_image = make_grid(tensor_grid, nrow=grid_size[1], normalize=True, padding=2)
    grid_image = ToPILImage()(grid_image)

    # Save the grid as an image
    grid_image.save(f'{filename}.png')


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

selected_features = get_images_from_different_classes(train_dataloader, label_map['myeloblast'], label_map['neutrophil_segmented'])

start_latent, end_latent = [get_latent_vector(feature.float().to(device),) for feature in selected_features]

interpolate_gif("vae_interpolation_grid", start_latent[0], end_latent[0], latent_dim=30, steps_per_dim=10, grid_size=(30, 10))

