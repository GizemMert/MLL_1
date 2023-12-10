import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model4 import VariationalAutoencodermodel4

plt.rcParams['figure.dpi'] = 200


def plot_reconstructed(vae_model, dim1=0, dim2=1, r0=(-3, 3), r1=(-3, 3), n=12, latent_dim=30):
    w = 128
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.zeros((1, latent_dim)).to(device)
            z[0, dim1] = x
            z[0, dim2] = y
            x_hat = vae_model.decoder(z)
            x_hat = vae_model.img_decoder(x_hat)
            x_hat = x_hat.reshape(w, w).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
    plt.axis('off')
    plt.savefig('reconstructed_images.png', bbox_inches='tight', pad_inches=0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoencodermodel4(latent_dim=30)
model_save_path = 'trained_model4cp2.pth'
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()

plot_reconstructed(model, dim1=0, dim2=1, r0=(-3, 3), r1=(-3, 3), n=12, latent_dim=30)

