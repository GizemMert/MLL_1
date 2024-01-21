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


def interpolate_single_dimension_grid(filename, representative_latent, latent_dim=30, steps_per_dim=10,
                                      grid_size=(30, 10)):
    model.eval()

    # We'll modify each dimension within a certain range around the representative latent vector
    # Here, we assume a range of -2 to 2 standard deviations if the latent space follows a standard normal distribution
    latent_range = 2
    step_size = latent_range * 2 / (steps_per_dim - 1)

    interpolated_images = []
    for dim in range(latent_dim):
        for step in range(steps_per_dim):
            # Create a variation only in the current dimension
            varied_latent = representative_latent.clone()
            varied_latent[dim] = varied_latent[dim] - latent_range + step * step_size
            varied_latent = varied_latent.to(device).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                # Use the two-step decoding process as per your model's structure
                intermediate_representation = model.decoder(varied_latent)
                decoded_img = model.img_decoder(intermediate_representation)

            # Append the decoded image to the list, make sure to remove the batch dimension
            interpolated_images.append(decoded_img.squeeze(0))

    # Convert list of tensors to a single tensor
    tensor_grid = torch.stack(interpolated_images)
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

single_class_feature = get_images_from_different_classes(train_dataloader, label_map['myeloblast'], label_map['myeloblast'])[0]
single_class_latent, _, _, _, _ = model(single_class_feature.float().to(device))

# Call the function with the correct parameters
interpolate_single_dimension_grid("vae_interpolation_single_class_grid", single_class_latent[0], latent_dim=30, steps_per_dim=10, grid_size=(30, 10))