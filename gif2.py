from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model4 import VariationalAutoencodermodel4
from Dataloader import Dataloader, label_map
from torch.utils.data import DataLoader


def interpolate_gif(model, filename, x_1, x_2, n=100, latent_dim=30):
    model.eval()

    # Encode the input images to latent vectors
    z_1, _ = model.encode(x_1.to(device))
    z_2, _ = model.encode(x_2.to(device))

    # Interpolate in the latent space
    z = torch.stack([z_1 + (z_2 - z_1) * t for t in np.linspace(0, 1, n)])

    # Decode the latent vectors
    interpolate_list = model.decoder(z)
    interpolate_list = model.img_decoder(interpolate_list)
    interpolate_list = interpolate_list.permute(0, 2, 3, 1).to('cpu').detach().numpy() * 255

    # Convert to PIL images and resize
    images_list = [Image.fromarray(img.astype(np.uint8)).resize((256, 256)) for img in interpolate_list]
    images_list = images_list + images_list[::-1]

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

# Extract two sample images from your dataloader
train_dataset = Dataloader(split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)

# Get two batches of data; x_1 and x_2 will be the first image from each batch
x_1, x_2 = None, None
for i, (_, scimg, _, _) in enumerate(train_dataloader):
    if i == 0:
        x_1 = scimg.to('cuda' if torch.cuda.is_available() else 'cpu')[0].unsqueeze(0)
    elif i == 1:
        x_2 = scimg.to('cuda' if torch.cuda.is_available() else 'cpu')[0].unsqueeze(0)
        break

# Ensure x_1 and x_2 are not None
if x_1 is not None and x_2 is not None:
    # Call the interpolate_gif function
    interpolate_gif(model, "vae_interpolation_2", x_1, x_2)
else:
    print("Error: Could not extract two images from the dataloader.")

