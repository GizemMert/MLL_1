import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from umap import UMAP

from Dataloader import Dataloader
from SSIM import SSIM
from model import Autoencodermodel

epochs = 150
batch_size = 128
ngpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencodermodel()
model_name = 'AE-CFE-'

if ngpu > 1:
    model = nn.DataParallel(model)

model = model.to(device)

# Load the dataset
dataset = Dataloader()
traindataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
criterion_1 = SSIM(window_size=10, size_average=True)

all_z = []
latent_space_dimension = None

umap_dir = 'umap_figures'
if not os.path.exists(umap_dir):
    os.makedirs(umap_dir)

for epoch in range(epochs):
    loss = 0
    acc_imrec_loss = 0
    acc_featrec_loss = 0

    model.train()

    for feat, scimg, label, _ in traindataloader:
        feat = feat.float()
        scimg = scimg.float()

        feat, scimg = feat.to(device), scimg.to(device)

        optimizer.zero_grad()

        z, output, im_out = model(feat)
        z_cpu = z.data.cpu().numpy()
        all_z.extend(z_cpu)
        im_out = im_out.squeeze()

        feat_rec_loss = criterion(output, feat)
        imrec_loss = 1 - criterion_1(im_out, scimg)
        train_loss = imrec_loss + feat_rec_loss

        train_loss.backward()
        optimizer.step()

        loss += train_loss.data.cpu()
        acc_featrec_loss += feat_rec_loss.data.cpu()
        acc_imrec_loss += imrec_loss.data.cpu()

    loss = loss / len(traindataloader)
    acc_featrec_loss = acc_featrec_loss / len(traindataloader)
    acc_imrec_loss = acc_imrec_loss / len(traindataloader)
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}, feat_loss = {:.6f},imrec_loss = {:.6f}".format
          (epoch + 1, epochs, loss, acc_featrec_loss, acc_imrec_loss))

    model.eval()

    if epoch % 10 == 0:

        if latent_space_dimension is None:
            z_shape = z.shape[1]  # Get the number of features in z
            latent_space_dimension = z_shape

        # UMAP for latent space
        latent_data = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean').fit_transform(
            np.vstack(all_z))

        plt.figure(figsize=(12, 10), dpi=150)
        scatter = plt.scatter(latent_data[:, 0], latent_data[:, 1], s=1, cmap='Spectral')
        plt.colorbar(scatter)
        plt.title('Latent Space Representation using UMAP', fontsize=18)
        plt.xlabel('UMAP Dimension 1', fontsize=14)
        plt.ylabel('UMAP Dimension 2', fontsize=14)

        umap_figure_filename = os.path.join(umap_dir, f'umap_epoch_{epoch}.png')

        # Save the UMAP figure
        plt.savefig(umap_figure_filename, dpi=300)
        plt.close()

    """
    model.eval()
    for i in range(50):
        ft, img, lbl, _, _ = dataset[i]
        ft = np.expand_dims(ft, axis=0)
        ft = torch.tensor(ft)
        ft = ft.to(device)

        _, _, im_out = model(ft)
        im_out = im_out.data.cpu().numpy()
        im_out = np.squeeze(im_out)
        im_out = np.moveaxis(im_out, 0, 2)
        img = np.moveaxis(img, 0, 2)
        im = np.concatenate([img, im_out], axis=1)

        if epoch % 10 == 0:
            file_name = "out-images/"
            if os.path.exists(os.path.join(file_name)) is False:
                os.makedirs(os.path.join(file_name))
            cv2.imwrite(os.path.join(file_name, str(i) + "-" + str(epoch) + ".jpg"), im * 255)
    """

if os.path.exists(os.path.join('Model/')) is False:
    os.makedirs(os.path.join('Model/'))
torch.save(model, "Model/" + model_name + time.strftime("%Y%m%d-%H%M%S") + ".mdl")
