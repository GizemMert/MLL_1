from PIL import Image
import torch
import numpy as np
from model4 import VariationalAutoencodermodel4, reparametrize
from Dataloader import Dataloader
from torch.utils.data import DataLoader
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
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


def interpolate_gif_with_gpr(model, filename, features, n=100, latent_dim=30):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def get_latent_vector(x):
        distributions = model.encoder(x)
        mu = distributions[:, :latent_dim]
        logvar = distributions[:, latent_dim:]
        z = reparametrize(mu, logvar)
        return z

    # Generate the latent representations
    latents = [get_latent_vector(feature.to(device)) for feature in features]

    # Prepare data for GPR
    X_train = np.arange(len(latents)).reshape(-1, 1)  # Indices
    Y_train = np.array([latent_vector.squeeze(0).cpu().detach().numpy() for latent_vector in latents])  # Latent vectors

    # Define a kernel for GPR
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1e3))

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
    gp.fit(X_train, Y_train)

    # predictions (interpolation)
    X_to_predict = np.linspace(0, len(latents) - 1, n).reshape(-1, 1)
    Y_pred = gp.predict(X_to_predict)

    # Decode interpolated latent vectors to images
    interpolate_list = []
    for z in Y_pred:
        z_tensor = torch.from_numpy(z).float().to(device).unsqueeze(0)
        y = model.decoder(z_tensor)
        img = model.img_decoder(y)
        img = img.permute(0, 2, 3, 1)
        interpolate_list.append(img.squeeze(0).to('cpu').detach().numpy())

    # Normalize and convert to PIL images, and save GIF
    interpolate_list = [np.clip(img * 255, 0, 255).astype(np.uint8) for img in interpolate_list]
    images_list = [Image.fromarray(img) for img in interpolate_list]
    images_list = images_list + images_list[::-1]
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
model_save_path = 'trained_model4cp2.pth'
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()


def get_images_from_different_classes(dataloader, class_1_label, class_2_label):
    feature_1, feature_2 = None, None

    for feature, _, labels, _ in dataloader:
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
selected_images = [feature.float().to(device) for feature in selected_features if feature is not None]

# Now, you can use these images for your interpolation GIF
interpolate_gif_with_gpr(model, "vae_interpolation_gpr", selected_images)
