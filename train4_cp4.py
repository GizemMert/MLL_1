import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from umap import UMAP
import torch.nn.functional as F
from sklearn.metrics import f1_score
from Dataloader import Dataloader, label_map
from SSIM import SSIM
from model4 import VariationalAutoencodermodel4
import os
import time
import cv2
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
# import mrcnn.config
# import mrcnn.model_feat_extract

"""
class SimpleConfig(mrcnn.config.Config):
    NAME = "march_mrcnn"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # Adjust based on your dataset



mask_rcnn_model = mrcnn.model_feat_extract.MaskRCNN(mode="inference",
                                                    config=SimpleConfig(),
                                                    model_dir=os.getcwd())
mask_rcnn_model.load_weights('/lustre/groups/aih/raheleh.salehi/MASKRCNN-STORAGE/MRCNN-leukocyte/logs/cells20220215T1028/mask_rcnn_cells_0004.h5', by_name=True)
"""

inverse_label_map = {v: k for k, v in label_map.items()}  # inverse mapping for UMAP
epochs = 300
batch_size = 128
ngpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(label_map)
model = VariationalAutoencodermodel4(latent_dim=30)
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

cff_feat_rec = 0.20
cff_im_rec = 0.45
cff_kld = 0.15
cff_edge = 0.20

beta = 4

umap_dir = 'umap_figures4cp2_new'
if not os.path.exists(umap_dir):
    os.makedirs(umap_dir)

latent_dir = 'latent_data4cp2_new'
if not os.path.exists(latent_dir):
    os.makedirs(latent_dir)

result_dir = "training_results4cp2_new"
os.makedirs(result_dir, exist_ok=True)
result_file = os.path.join(result_dir, "training_results4cp2_new.txt")


def kl_divergence(mu, logvar):
    batch_s = mu.size(0)
    assert batch_s != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def reconstruction_loss(scimg, im_out, distribution="gaussian"):
    batch_s = scimg.size(0)
    assert batch_s != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(im_out, scimg, reduction="sum").div(batch_s)
    elif distribution == 'gaussian':
        recon_loss = F.mse_loss(im_out, scimg, reduction='sum').div(batch_size)
    else:
        recon_loss = None

    return recon_loss

class SobelFilter(nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.weight_x = nn.Parameter(data=sobel_kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=sobel_kernel_y, requires_grad=False)

    def forward(self, x):
        x_gray = torch.mean(x, dim=1, keepdim=True)
        edge_x = F.conv2d(x_gray, self.weight_x, padding=1)
        edge_y = F.conv2d(x_gray, self.weight_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

        return edge

edge_loss_fn = SobelFilter().to(device)


for epoch in range(epochs):
    loss = 0
    acc_imrec_loss = 0
    acc_featrec_loss = 0
    kl_div_loss = 0
    y_true = []
    y_pred = []

    model.train()

    if epoch % 10 == 0:
        all_means = []
        all_labels = []

    for feat, scimg, label, mask, _ in train_dataloader:
        feat = feat.float()
        scimg = scimg.float()
        label = label.long().to(device)
        mask = mask.float().to(device)

        """
        np_scimg = scimg.permute(0, 2, 3, 1).cpu().numpy()
        batch_masks = []
        sub_batch_size = 16  # Adjust as per your GPU capability
        for i in range(0, len(np_scimg), sub_batch_size):
            sub_images = np_scimg[i:i + sub_batch_size]
            results = mask_rcnn_model.detect(sub_images)
            sub_masks = [r['masks'] for r in results if r]
            batch_masks.extend(sub_masks)

        np_masks = np.concatenate(batch_masks, axis=0) if batch_masks else np.zeros_like(np_scimg)
        masks = torch.from_numpy(np_masks).float().to(device)
        """

        feat, scimg = feat.to(device), scimg.to(device)

        optimizer.zero_grad()

        z_dist, output, im_out, mu, logvar = model(feat)
        masked_scimg = scimg * mask.unsqueeze(1)

        imgs_edges = edge_loss_fn(masked_scimg)
        recon_edges = edge_loss_fn(im_out)

        edge_loss = F.mse_loss(recon_edges, imgs_edges)
        feat_rec_loss = criterion(output, feat)
        recon_loss = reconstruction_loss(masked_scimg, im_out, distribution="gaussian")
        kld_loss, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        train_loss = (cff_feat_rec * feat_rec_loss) + (cff_im_rec * recon_loss) + (cff_kld * kld_loss) + (cff_edge * edge_loss)

        train_loss.backward()
        optimizer.step()

        loss += train_loss.data.cpu()
        acc_featrec_loss += feat_rec_loss.data.cpu()
        acc_imrec_loss += recon_loss.data.cpu()
        kl_div_loss += kld_loss.data.cpu()

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
          (epoch + 1, epochs, loss.item(), acc_featrec_loss.item(), acc_imrec_loss.item(), kl_div_loss.item()))

    with open(result_file, "a") as f:
        f.write(f"Epoch {epoch + 1}: Loss = {loss.item():.6f}, Feat_Loss = {acc_featrec_loss.item():.6f}, "
                f"Img_Rec_Loss = {acc_imrec_loss.item():.6f}, KL_DIV = {kl_div_loss.item():.6f} \n")

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

        # Filter out the 'erythroblast' class
        erythroblast_class_index = label_map['erythroblast']
        mask = all_labels_array != erythroblast_class_index
        filtered_latent_data = latent_data_reshaped[mask]
        filtered_labels = all_labels_array[mask]

        # UMAP for latent space
        latent_data_umap = UMAP(n_neighbors=13, min_dist=0.1, n_components=2, metric='euclidean').fit_transform(
            filtered_latent_data)

        fig = plt.figure(figsize=(12, 10), dpi=150)
        gs = GridSpec(1, 2, width_ratios=[4, 1], figure=fig)

        ax = fig.add_subplot(gs[0])
        scatter = ax.scatter(latent_data_umap[:, 0], latent_data_umap[:, 1], s=100, c=filtered_labels, cmap='Spectral', edgecolor=(1, 1, 1, 0.7))
        ax.set_aspect('equal')

        x_min, x_max = np.min(latent_data_umap[:, 0]), np.max(latent_data_umap[:, 0])
        y_min, y_max = np.min(latent_data_umap[:, 1]), np.max(latent_data_umap[:, 1])

        zoom_factor = 0.40  # Smaller values mean more zoom
        padding_factor = 0.3  # Adjust padding around the zoomed area

        # Calculate the range for zooming in based on the zoom factor
        x_range = (x_max - x_min) * zoom_factor
        y_range = (y_max - y_min) * zoom_factor

        # Calculate the center of the data
        center_x = (x_max + x_min) / 2
        center_y = (y_max + y_min) / 2

        # Calculate new limits around the center of the data
        new_x_min = center_x - (x_range * (1 + padding_factor))
        new_x_max = center_x + (x_range * (1 + padding_factor))
        new_y_min = center_y - (y_range * (1 + padding_factor))
        new_y_max = center_y + (y_range * (1 + padding_factor))

        # Apply the new limits to zoom in on the plot
        ax.set_xlim(new_x_min, new_x_max)
        ax.set_ylim(new_y_min, new_y_max)

        ax.set_title(f'Latent Space Representation - (Epoch {epoch})', fontsize=18)
        ax.set_xlabel('UMAP Dimension 1', fontsize=16)
        ax.set_ylabel('UMAP Dimension 2', fontsize=16)

        # Second subplot for the legend
        ax_legend = fig.add_subplot(gs[1])
        ax_legend.axis('off')  # Turn off the axis for the legend subplot

        unique_filtered_labels = np.unique(filtered_labels)
        filtered_class_names = [inverse_label_map[label] for label in unique_filtered_labels if label in inverse_label_map]
        color_map = plt.cm.Spectral(np.linspace(0, 1, len(unique_filtered_labels)))

        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=filtered_class_names[i],
                                     markerfacecolor=color_map[i], markersize=18) for i in range(len(filtered_class_names))]

        ax_legend.legend(handles=legend_handles, loc='center', fontsize=16, title='Cell Types')

        plt.tight_layout()
        umap_figure_filename = os.path.join(umap_dir, f'umap_epoch_{epoch}.png')
        plt.savefig(umap_figure_filename, bbox_inches='tight', dpi=300)
        plt.close(fig)

    for i in range(30):
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
            file_name = "reconsructed-images4_cp2_new/"
            if os.path.exists(os.path.join(file_name)) is False:
                os.makedirs(os.path.join(file_name))
            cv2.imwrite(os.path.join(file_name, str(i) + "-" + str(epoch) + ".jpg"), im * 255)

script_dir = os.path.dirname(__file__)

model_save_path = os.path.join(script_dir, 'trained_model4cp2_new.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Trained model saved to {model_save_path}")

with open(result_file, "a") as f:
    f.write("Training completed.\n")

""""
if os.path.exists(os.path.join('Model/')) is False:
    os.makedirs(os.path.join('Model/'))
torch.save(model, "Model/" + model_name + time.strftime("%Y%m%d-%H%M%S") + ".mdl")
"""