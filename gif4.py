from model4 import VariationalAutoencodermodel4, reparametrize
import torch
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained model
model = VariationalAutoencodermodel4(latent_dim=30)  # Initialize with appropriate parameters
model_save_path = 'trained_model4cp2.pth'
model.load_state_dict(torch.load(model_save_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
model.to(device)
model.eval()


def sample_latent_points(latent_dim, num_samples=100):
    mu = torch.randn(num_samples, latent_dim).to(device)
    logvar = torch.randn(num_samples, latent_dim).to(device)
    z = reparametrize(mu, logvar)
    return z


def generate_images(model, latent_points):
    with torch.no_grad():
        _, y, images, _, _ = model(latent_points)
    return images


latent_dim = 30
latent_points = sample_latent_points(latent_dim, num_samples=100)
generated_images = generate_images(model, latent_points)


generated_images = (generated_images.cpu().numpy() * 255).astype(np.uint8)
generated_images = [Image.fromarray(img.transpose(1, 2, 0)) for img in generated_images]


gif_path = 'vae_gif4.gif'
imageio.mimsave(gif_path, generated_images, fps=5)
