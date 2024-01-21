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


def interpolate_gif(filename, latents, latent_dim=30, grid_size=(5, 50)):
    model.eval()

    def interpolate_single_dimension(start, end, dim, n=100):
        # Generate interpolation for a single dimension
        interpolation_path = []
        for t in torch.linspace(0, 1, steps=n, device=device):
            point = start.clone()
            point[0][dim] = (1 - t) * start[0][dim] + t * end[0][dim]
            interpolation_path.append(point)
        return interpolation_path

    start_latent, end_latent = latents
    all_interpolations = []

    # Interpolate each dimension separately
    for dim in range(latent_dim):
        all_interpolations.extend(interpolate_single_dimension(start_latent, end_latent, dim, n=100))

    interpolate_tensors = []
    for z in all_interpolations:
        with torch.no_grad():
            z = z.to(device)
            img = model.decoder(z)
            img = model.img_decoder(img)
            img = img.squeeze(0)  # Assuming the output is (1, C, H, W)
            interpolate_tensors.append(img.cpu())  # Keep it as a tensor

    # Make sure you have the correct number of images to fill the grid
    while len(interpolate_tensors) < grid_size[0] * grid_size[1]:
        interpolate_tensors.append(torch.zeros_like(interpolate_tensors[0]))

    # Convert list of tensors to a single tensor
    tensor_grid = torch.stack(interpolate_tensors)
    # Create a grid of images
    image_grid = make_grid(tensor_grid, nrow=grid_size[1], normalize=True)
    # Convert the grid to a PIL Image
    grid_image = ToPILImage()(image_grid)
    # Save the grid as an image
    grid_image.save(f'{filename}_dimension_wise.png')


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

# Convert to appropriate format and device
start_latent, end_latent = [get_latent_vector(feature.float().to(device),) for feature in selected_features]

# Now, you can use these images for your interpolation GIF
interpolate_gif("vae_interpolation_grid_dimension_wise", [start_latent, end_latent], latent_dim=30, grid_size=(5, 30))
