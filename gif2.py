from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model4 import VariationalAutoencodermodel4, reparametrize
from Dataloader import Dataloader, label_map
from torch.utils.data import DataLoader


def interpolate_gif(model, filename, features, n=100, latent_dim=30):
    model.eval()

    # Function to extract the latent vector from the model output
    def get_latent_vector(x):
        distributions = model.encoder(x)
        mu = distributions[:, :latent_dim]
        logvar = distributions[:, latent_dim:]
        z = reparametrize(mu, logvar)
        return z

    latents = [get_latent_vector(feature.to(device)) for feature in features]

    all_interpolations = []
    for i in range(len(latents) - 1):
        z1, z2 = latents[i], latents[i + 1]
        # Generate interpolated latent vectors
        for t in np.linspace(0, 1, n):
            z_interp = z1 * (1 - t) + z2 * t
            all_interpolations.append(z_interp)

    # Decode the interpolated latent vectors
    interpolate_list = [model.decoder(z).squeeze(0) for z in all_interpolations]
    interpolate_list = [model.img_decoder(z).squeeze(0) for z in interpolate_list]
    interpolate_list = [z.permute(1, 2, 0).to('cpu').detach().numpy() * 255 for z in interpolate_list]

    # Convert to PIL images and resize
    images_list = [Image.fromarray(img.astype(np.uint8)).resize((256, 256)) for img in interpolate_list]
    images_list = images_list + images_list[::-1]  # Loop back to the beginning

    # Save as a GIF
    images_list[0].save(
        f'{filename}.gif',
        save_all=True,
        append_images=images_list[1:],
        loop=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load your model
model = VariationalAutoencodermodel4(latent_dim=30)
model_save_path = 'trained_model4cp2.pth'
model.load_state_dict(torch.load(model_save_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()


train_dataset = Dataloader(split='train')
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)

# Assuming the dataloader returns features and images
f_1, f_2 = None, None
for i, (features, _, _, _) in enumerate(train_dataloader):
    if i == 0:
        f_1 = features.float().to(device)[0].unsqueeze(0)  # Get the first feature tensor
    elif i == 1:
        f_2 = features.float().to(device)[0].unsqueeze(0)  # Get the second feature tensor
        break

if f_1 is not None and f_2 is not None:
    interpolate_gif(model, "vae_interpolation_2", [f_1, f_2])
else:
    print("Error: Could not extract two feature sets from the dataloader.")

