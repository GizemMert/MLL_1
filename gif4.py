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

def manhattan_interpolate(start, end, num_steps=10):
    diff = end - start
    steps = diff / num_steps
    current = start.clone()

    interpolations = [current.clone()]
    for dim in range(len(diff)):
        for _ in range(num_steps):
            current[dim] += steps[dim]
            interpolations.append(current.clone())

    return interpolations

def get_latent_vector(x, latent_dim=30):
    distributions = model.encoder(x)
    mu = distributions[:, :latent_dim]
    logvar = distributions[:, latent_dim:]
    z = reparametrize(mu, logvar)
    return z


def interpolate_gif_manhattan(filename, start_latent, end_latent, latent_dim=30, steps_per_dim=10, grid_size=(30, 10)):
    model.eval()

    # Generate the Manhattan path
    manhattan_path = manhattan_interpolate(start_latent.squeeze(), end_latent.squeeze(), steps_per_dim)

    decoded_images = []
    for z in manhattan_path:
        z = z.to(device).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            decoded_img = model.decoder(z)
            decoded_img = model.img_decoder(decoded_img)  # This should output the reconstructed image
        decoded_images.append(decoded_img)

    # Adjust the number of images to match the grid size
    while len(decoded_images) < grid_size[0] * grid_size[1]:
        decoded_images.append(torch.zeros_like(decoded_images[0]))

    decoded_images = decoded_images[:grid_size[0] * grid_size[1]]

    # Convert list of tensors to a single tensor
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

interpolate_gif_manhattan("vae_interpolation_grid_manhattan", start_latent[0], end_latent[0], latent_dim=30, steps_per_dim=10, grid_size=(30, 10))

