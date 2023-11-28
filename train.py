import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from umap import UMAP
from sklearn.metrics import f1_score
from Dataloader import Dataloader, label_map
from SSIM import SSIM
from model import VariationalAutoencodermodel

inverse_label_map = {v: k for k, v in label_map.items()}  # inverse mapping for UMAP
epochs = 150
batch_size = 128
ngpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(label_map)
model = VariationalAutoencodermodel(latent_dim=50)
model_name = 'AE-CFE-'

if ngpu > 1:
    model = nn.DataParallel(model)

model = model.to(device)

# Load the dataset
train_dataset = Dataloader(split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

criterion = nn.MSELoss()
criterion_1 = SSIM(window_size=10, size_average=True)
class_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

cff_feat_rec = 0.30
cff_im_rec = 0.40
cff_kl = 0.30

umap_dir = 'umap_figures'
if not os.path.exists(umap_dir):
    os.makedirs(umap_dir)

latent_dir = 'latent_data'
if not os.path.exists(latent_dir):
    os.makedirs(latent_dir)

result_dir = "training_results"
os.makedirs(result_dir, exist_ok=True)
result_file = os.path.join(result_dir, "training_results.txt")

for epoch in range(epochs):
    loss = 0
    acc_imrec_loss = 0
    acc_featrec_loss = 0
    kl_div_loss = 0
    # y_true = []
    # y_pred = []

    model.train()

    if epoch % 10 == 0:
        # all_latent_representations = []
        all_labels = []
        all_means = [] # for UMAP

    for feat, scimg, label, _ in train_dataloader:
        feat = feat.float()
        scimg = scimg.float()
        label = label.long().to(device)

        feat, scimg = feat.to(device), scimg.to(device)

        optimizer.zero_grad()

        z_dist, output, im_out, mu, logvar = model(feat)

        feat_rec_loss = criterion(output, feat)
        imrec_loss = 1 - criterion_1(im_out, scimg)
        #KL Divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # classification_loss = class_criterion(class_pred, label)
        train_loss = feat_rec_loss + imrec_loss + kl_div
        # (cff_class*classification_loss)

        train_loss.backward()
        optimizer.step()

        loss += train_loss.data.cpu()
        acc_featrec_loss += feat_rec_loss.data.cpu()
        acc_imrec_loss += imrec_loss.data.cpu()
        kl_div_loss += kl_div.data.cpu()

        if epoch % 10 == 0:
           all_means.append(mu.data.cpu().numpy())
           all_labels.extend(label.cpu().numpy())

        # y_true.extend(label.cpu().numpy())
        # _, predicted = torch.max(class_pred.data, 1)
        # y_pred.extend(predicted.cpu().numpy())

    loss = loss / len(train_dataloader)
    acc_featrec_loss = acc_featrec_loss / len(train_dataloader)
    acc_imrec_loss = acc_imrec_loss / len(train_dataloader)
    kl_div_loss = kl_div_loss / len(train_dataloader)
    # f1 = f1_score(y_true, y_pred, average='weighted')

    print("epoch : {}/{}, loss = {:.6f}, feat_loss = {:.6f}, imrec_loss = {:.6f}, kl_div = {:.6f}".format
          (epoch + 1, epochs, loss, acc_featrec_loss, acc_imrec_loss, kl_div_loss))

    with open(result_file, "a") as f:
        f.write(f"Epoch {epoch + 1}: Loss = {loss:.6f}, Feat_Loss = {acc_featrec_loss:.6f}, "
                f"Img_Rec_Loss = {acc_imrec_loss:.6f}, KL_DIV = {kl_div_loss:.6f} \n")

    if epoch % 10 == 0:
        latent_filename = os.path.join(latent_dir, f'latent_epoch_{epoch}.npy')
        np.save(latent_filename, np.concatenate(all_means, axis=0))

    model.eval()

    if epoch % 10 == 0:
        # Load all latent representations
        latent_data = np.load(latent_filename)
        latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
        print(latent_data_reshaped.shape)
        all_labels_array = np.array(all_labels)
        # print("Labels array shape:", all_labels_array.shape)

        original_labels = [inverse_label_map[label] for label in all_labels_array]

        # UMAP for latent space
        latent_data_umap = UMAP(n_neighbors=13, min_dist=0.1, n_components=2, metric='euclidean').fit_transform(
            latent_data_reshaped)

        plt.figure(figsize=(12, 10), dpi=150)
        scatter = plt.scatter(latent_data_umap[:, 0], latent_data_umap[:, 1], s=1, c=all_labels_array, cmap='Spectral')

        color_map = plt.cm.Spectral(np.linspace(0, 1, len(set(all_labels_array))))
        class_names = [inverse_label_map[i] for i in range(len(inverse_label_map))]

        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[i],
                                     markerfacecolor=color_map[i], markersize=10) for i in range(len(class_names))]
        plt.legend(handles=legend_handles, loc='lower right', title='Cell Types')

        plt.title(f'Latent Space Representation - (Epoch {epoch})', fontsize=18)
        plt.xlabel('UMAP Dimension 1', fontsize=14)
        plt.ylabel('UMAP Dimension 2', fontsize=14)

        umap_figure_filename = os.path.join(umap_dir, f'umap_epoch_{epoch}.png')

        # Save the UMAP figure
        plt.savefig(umap_figure_filename, dpi=300)
        plt.close()

    for i in range(50):
        ft, img, lbl, _ = train_dataset[i]
        ft = np.expand_dims(ft, axis=0)
        ft = torch.tensor(ft)
        ft = ft.to(device)

        _, _, im_out, _, _ = model(ft)
        im_out = im_out.data.cpu().numpy()
        im_out = np.squeeze(im_out)
        im_out = np.moveaxis(im_out, 0, 2)
        img = np.moveaxis(img, 0, 2)
        im = np.concatenate([img, im_out], axis=1)

        if epoch % 10 == 0:
            file_name = "reconsructed-images/"
            if os.path.exists(os.path.join(file_name)) is False:
                os.makedirs(os.path.join(file_name))
            cv2.imwrite(os.path.join(file_name, str(i) + "-" + str(epoch) + ".jpg"), im * 255)

script_dir = os.path.dirname(__file__)

model_save_path = os.path.join(script_dir, 'trained_model.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Trained model saved to {model_save_path}")

with open(result_file, "a") as f:
    f.write("Training completed.\n")

""""
if os.path.exists(os.path.join('Model/')) is False:
    os.makedirs(os.path.join('Model/'))
torch.save(model, "Model/" + model_name + time.strftime("%Y%m%d-%H%M%S") + ".mdl")
"""
