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
from sklearn.preprocessing import LabelEncoder


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

label_encoder = LabelEncoder()
all_categorical_labels = dataset.get_all_labels()  # You need to implement this method in your Dataloader
label_encoder.fit(all_categorical_labels)

umap_dir = 'umap_figures'
if not os.path.exists(umap_dir):
    os.makedirs(umap_dir)

latent_dir = 'latent_data'
if not os.path.exists(latent_dir):
    os.makedirs(latent_dir)

for epoch in range(epochs):
    loss = 0
    acc_imrec_loss = 0
    acc_featrec_loss = 0

    model.train()

    if epoch % 10 == 0:
        all_latent_representations = []
        all_labels = []

    for feat, scimg, label, _ in traindataloader:
        feat = feat.float()
        scimg = scimg.float()
        numeric_labels = label_encoder.transform(label)  # Convert to int, assuming one label per sample

        feat, scimg = feat.to(device), scimg.to(device)

        optimizer.zero_grad()

        z, output, im_out = model(feat)

        feat_rec_loss = criterion(output, feat)
        imrec_loss = 1 - criterion_1(im_out, scimg)
        train_loss = imrec_loss + feat_rec_loss

        train_loss.backward()
        optimizer.step()

        loss += train_loss.data.cpu()
        acc_featrec_loss += feat_rec_loss.data.cpu()
        acc_imrec_loss += imrec_loss.data.cpu()

        if epoch % 10 == 0:
            all_latent_representations.append(z.data.cpu().numpy())
            all_labels.extend(numeric_labels)

    loss = loss / len(traindataloader)
    acc_featrec_loss = acc_featrec_loss / len(traindataloader)
    acc_imrec_loss = acc_imrec_loss / len(traindataloader)
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}, feat_loss = {:.6f},imrec_loss = {:.6f}".format
          (epoch + 1, epochs, loss, acc_featrec_loss, acc_imrec_loss))

    if epoch % 10 == 0:
        latent_filename = os.path.join(latent_dir, f'latent_epoch_{epoch}.npy')
        np.save(latent_filename, np.concatenate(all_latent_representations, axis=0))

    model.eval()

    if epoch % 10 == 0:
        # Load all latent representations from saved files
        latent_data = np.load(latent_filename)
        latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
        print(latent_data_reshaped.shape)
        all_labels_array = np.array(all_labels)
        # print("Labels array shape:", all_labels_array.shape)
        # print("Labels array dtype:", all_labels_array.dtype)

        original_labels = label_encoder.inverse_transform(all_labels_array)

        # UMAP for latent space
        latent_data_umap = UMAP(n_neighbors=13, min_dist=0.05, n_components=2, metric='euclidean').fit_transform(
            latent_data_reshaped)

        plt.figure(figsize=(12, 10), dpi=150)
        scatter = plt.scatter(latent_data_umap[:, 0], latent_data_umap[:, 1], s=1, c=all_labels_array, cmap='Spectral')

        for i, txt in enumerate(original_labels):
            plt.annotate(txt, (latent_data_umap[i, 0], latent_data_umap[i, 1]), fontsize=8, ha='right', va='bottom')

        plt.colorbar(scatter)
        plt.title(f'Latent Space Representation - (Epoch {epoch})', fontsize=18)
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
