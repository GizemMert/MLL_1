from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model4 import VariationalAutoencodermodel4
from Dataloader import Dataloader, label_map
from torch.utils.data import DataLoader


def interpolate_gif(model, filename, imgs, n=100, latent_dim=30):
    model.eval()
    latents = []

    # Encode the images to latent vectors
    for img in imgs:
        z, _ = model.encode(img.to(device))
        latents.append(z)

    # Interpolate in the latent space between each pair of images
    all_interpolations = []
    for i in range(len(latents) - 1):
        z1 = latents[i]
        z2 = latents[i + 1]
        interpolations = [z1 + (z2 - z1) * t for t in np.linspace(0, 1, n)]
        all_interpolations.extend(interpolations)

    # Decode the latent vectors
    z = torch.stack(all_interpolations)
    interpolate_list = model.decoder(z)
    interpolate_list = model.img_decoder(interpolate_list)
    interpolate_list = interpolate_list.permute(0, 2, 3, 1).to('cpu').detach().numpy() * 255

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


def get_images_from_different_classes(dataloader, num_classes=4):
    images_from_classes = {}
    for _, image, label, _ in dataloader:
        if label.item() not in images_from_classes and len(images_from_classes) < num_classes:
            images_from_classes[label.item()] = image

        if len(images_from_classes) == num_classes:
            break

    return list(images_from_classes.values())


# Extract two sample images from your dataloader
train_dataset = Dataloader(split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)

selected_images = get_images_from_different_classes(train_dataloader)

# Convert to appropriate format and device
selected_images = [img.to(device) for img in selected_images]

# Now, you can use these images for your interpolation GIF
interpolate_gif(model, "vae_interpolation_4", selected_images)