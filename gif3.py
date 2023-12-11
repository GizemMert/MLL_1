from PIL import Image
import torch
import numpy as np
from model4 import VariationalAutoencodermodel4, reparametrize
from Dataloader import Dataloader
from torch.utils.data import DataLoader

def interpolate_gif(model, filename, features, n=100, latent_dim=30):
    model.eval()

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Function to extract the latent vector from the model output
    def get_latent_vector(x):
        distributions = model.encoder(x)
        mu = distributions[:, :latent_dim]
        logvar = distributions[:, latent_dim:]
        z = reparametrize(mu, logvar)
        return z

    # Generate the latent representations
    latents = [get_latent_vector(feature.to(device)) for feature in features]

    # Interpolate between the latent vectors
    all_interpolations = []
    for i in range(len(latents) - 1):
        for t in np.linspace(0, 1, n):
            z_interp = latents[i] * (1 - t) + latents[i + 1] * t
            all_interpolations.append(z_interp)

    # Decode the latent vectors and generate images
    interpolate_list = []
    for z in all_interpolations:
        y = model.decoder(z)
        img = model.img_decoder(y)
        img = img.permute(0, 2, 3, 1)  # Assuming the output is [batch_size, channels, height, width]
        interpolate_list.append(img.squeeze(0).to('cpu').detach().numpy())

    # Normalize the pixel values to be between 0 and 255 for image saving
    interpolate_list = [np.clip(img * 255, 0, 255).astype(np.uint8) for img in interpolate_list]

    # Convert to PIL images and resize
    images_list = [Image.fromarray(img) for img in interpolate_list]
    images_list = images_list + images_list[::-1]  # Loop back to the beginning

    # Save as a GIF with a duration between frames of 100ms
    images_list[0].save(
        f'{filename}.gif',
        save_all=True,
        append_images=images_list[1:],
        loop=0,  # Loop forever
        duration=100  # Duration between frames in milliseconds
    )

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoencodermodel4(latent_dim=30)
model_save_path = 'trained_model4cp2.pth'
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()

# Extract features from different classes
train_dataset = Dataloader(split='train')
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)


def get_images_from_different_classes(dataloader, num_classes=13):
    features_from_classes = {}
    for feature, _, labels, _ in dataloader:
        for i, label in enumerate(labels):
            label_item = label.item()
            if label_item not in features_from_classes and len(features_from_classes) < num_classes:
                # Select the feature corresponding to the label
                selected_feature = feature[i].unsqueeze(0)
                features_from_classes[label_item] = selected_feature

            if len(features_from_classes) == num_classes:
                break

        if len(features_from_classes) == num_classes:
            break

    return list(features_from_classes.values())



# Extract two sample images from your dataloader
train_dataset = Dataloader(split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)

selected_features = get_images_from_different_classes(train_dataloader)

# Convert to appropriate format and device
selected_images = [feature.float().to(device) for feature in selected_features]

# Main loop to generate multiple GIFs
for i in range(10):
    filename = f"vae_interpolation_13_{i}"  # Unique filename for each iteration
    interpolate_gif(model, filename, selected_images)
