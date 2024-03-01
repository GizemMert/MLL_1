import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from umap import UMAP
import torch.nn.functional as F
from Dataloader_4 import Dataloader, label_map
from SSIM import SSIM
from model4 import VariationalAutoencodermodel4
import os
import cv2
from mmd_loss_2 import mmd
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np

label_map = {
    'basophil': 0,
    'eosinophil': 1,
    'erythroblast': 2,
    'myeloblast': 3,
    'promyelocyte': 4,
    'myelocyte': 5,
    'metamyelocyte': 6,
    'neutrophil_banded': 7,
    'neutrophil_segmented': 8,
    'monocyte': 9,
    'lymphocyte_typical': 10,
    'lymphocyte_atypical': 11,
    'smudge_cell': 12,
}

inverse_label_map = {v: k for k, v in label_map.items()}  # inverse mapping for UMAP
epochs = 290
batch_size = 128
ngpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(label_map)
model = VariationalAutoencodermodel4(latent_dim=50)
model_name = 'AE-CFE-'

epoch_of_gen = 290
latent_dir_ge = 'latent_variables_GE_3_cp'
z_dir_ge = 'z_variables_GE_3_cp'

class_labels_gen = [0, 1, 2, 3, 4, 5]
# {"label_mapping": {"monocyte": 0, "neutrophil_Liver": 1, "neutrophil_Blood": 2, "basophil": 3, "myeloid cell": 4, "neutrophil_Lung": 5}

class_label_n_blood = 2
class_label_n_liver = 1
class_label_n_lung = 5
class_label_m = 0
class_label_myeloid = 4


mean_filename_n_blood = os.path.join(latent_dir_ge, f'class_{class_label_n_blood}_mean_epoch_{epoch_of_gen}.npy')
z_filename_n_blood = os.path.join(z_dir_ge, f'class_{class_label_n_blood}_z_epoch_{epoch_of_gen}.npy')

mean_filename_n_liver = os.path.join(latent_dir_ge, f'class_{class_label_n_liver}_mean_epoch_{epoch_of_gen}.npy')
z_filename_n_liver = os.path.join(z_dir_ge, f'class_{class_label_n_liver}_z_epoch_{epoch_of_gen}.npy')

mean_filename_n_lung = os.path.join(latent_dir_ge, f'class_{class_label_n_lung}_mean_epoch_{epoch_of_gen}.npy')
z_filename_n_lung = os.path.join(z_dir_ge, f'class_{class_label_n_lung}_z_epoch_{epoch_of_gen}.npy')

mean_filename_m = os.path.join(latent_dir_ge, f'class_{class_label_m}_mean_epoch_{epoch_of_gen}.npy')
z_filename_m = os.path.join(z_dir_ge, f'class_{class_label_m}_z_epoch_{epoch_of_gen}.npy')

mean_filename_myeloid = os.path.join(latent_dir_ge, f'class_{class_label_myeloid}_mean_epoch_{epoch_of_gen}.npy')
z_filename_myeloid = os.path.join(z_dir_ge, f'class_{class_label_myeloid}_z_epoch_{epoch_of_gen}.npy')

ref_mean_class_n_blood = torch.from_numpy(np.load(mean_filename_n_blood)).float().to(device)
ref_z_class_n_blood = torch.from_numpy(np.load(z_filename_n_blood)).float().to(device)

ref_mean_class_mono = torch.from_numpy(np.load(mean_filename_m)).float().to(device)
ref_z_class_mono = torch.from_numpy(np.load(z_filename_m)).float().to(device)

ref_mean_class_myelo = torch.from_numpy(np.load(mean_filename_myeloid)).float().to(device)
ref_z_class_myelo = torch.from_numpy(np.load(z_filename_myeloid)).float().to(device)

ref_mean_class_n_liver = torch.from_numpy(np.load(mean_filename_n_liver)).float().to(device)
ref_z_class_n_liver = torch.from_numpy(np.load(z_filename_n_liver)).float().to(device)

ref_mean_class_n_lung = torch.from_numpy(np.load(mean_filename_n_lung)).float().to(device)
ref_z_class_n_lung = torch.from_numpy(np.load(z_filename_n_lung)).float().to(device)

if ngpu > 1:
    model = nn.DataParallel(model)

model = model.to(device)

# Load the dataset
train_dataset = Dataloader(split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

criterion = nn.MSELoss()
criterion_1 = SSIM(window_size=10, size_average=True)
class_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)


# custom_weights_path = "/lustre/groups/aih/raheleh.salehi/MASKRCNN-STORAGE/MRCNN-leukocyte/logs/cells20220215T1028/mask_rcnn_cells_0004.h5"
# custom_state_dict = torch.load(custom_weights_path)
# mask_rcnn_model.load_state_dict(custom_state_dict)

cff_feat_rec = 0.05
cff_im_rec = 0.20
cff_kld = 0.05
cff_mmd_n_blood = 0.30
cff_mmd_n_liver = 0.20
cff_mmd_n_lung = 50


beta = 4

umap_dir = 'umap_figures4cp2_new5_std_gen_2_cp'
if not os.path.exists(umap_dir):
    os.makedirs(umap_dir)

latent_dir = 'latent_data4cp2_new5_std_gen_2_cp'
if not os.path.exists(latent_dir):
    os.makedirs(latent_dir)

z_dir = 'z_data4cp2_new5_std_gen_2_cp'
if not os.path.exists(z_dir):
    os.makedirs(z_dir)

neutrophil_banded_z_dir = 'banded_z_data4cp2_new5_std_gen_2_cp'
if not os.path.exists(neutrophil_banded_z_dir):
    os.makedirs(neutrophil_banded_z_dir)

neutrophil_segment_z_dir = 'segment_z_data4cp2_new5_std_gen_2_cp'
if not os.path.exists(neutrophil_segment_z_dir):
    os.makedirs(neutrophil_segment_z_dir)

monocyte_z_dir = 'mono_z_data4cp2_new5_std_gen_2_cp'
if not os.path.exists(monocyte_z_dir):
    os.makedirs(monocyte_z_dir)

myle_z_dir = 'myle_z_data4cp2_new5_std_gen_2_cp'
if not os.path.exists(myle_z_dir):
    os.makedirs(myle_z_dir)

log_dir = 'log_data4cp2_new5_std_gen_2_cp'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

label_dir = 'label_data4cp2_new5_std_gen_2_cp'
if not os.path.exists(label_dir):
    os.makedirs(label_dir)

result_dir = "training_results4cp2_new5_std_gen_2_cp"
os.makedirs(result_dir, exist_ok=True)
result_file = os.path.join(result_dir, "training_results4cp2_new5_std_gen_2_cp.txt")

save_img_dir = "masked_images5_std_gen_2_cp"
if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir)

save_mask_dir = "masks5_std_gen_2_cp"
if not os.path.exists(save_mask_dir):
    os.makedirs(save_mask_dir)


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
ref_z_class_n_blood = ref_z_class_n_blood.to(device)
ref_z_class_n_liver = ref_z_class_n_liver.to(device)
ref_z_class_n_lung = ref_z_class_n_lung.to(device)
ref_z_class_mono = ref_z_class_mono.to(device)
ref_z_class_myelo = ref_z_class_myelo.to(device)

for epoch in range(epochs):
    loss = 0.0
    acc_imrec_loss = 0.0
    acc_featrec_loss = 0.0
    kl_div_loss = 0.0
    mmd_loss_n_blood = 0.0
    mmd_loss_n_liver = 0.0
    mmd_loss_n_lung = 0.0
    y_true = []
    y_pred = []
    count_neutrophil_banded = 0
    count_neutrophil_segmented = 0
    # neutrophil_z_vectors = []

    model.train()

    if epoch % 10 == 0:
        all_means = []
        all_labels = []
        all_logvars = []
        all_z =[]
        all_z_n_band = []
        all_z_n_segmented =[]


    for feat, scimg, mask, label, _ in train_dataloader:
        feat = feat.float()
        scimg = scimg.float()
        label = label.long().to(device)
        mask = mask.float().to(device)
        # mmd_loss_neutrophil = torch.tensor(0.0).to(device)
        # mmd_loss_monocyte = torch.tensor(0.0).to(device)
        # mmd_loss_myle = torch.tensor(0.0).to(device)
        count_neutrophil_banded += (label == label_map['neutrophil_banded']).sum().item()
        count_neutrophil_segmented += (label == label_map['neutrophil_segmented']).sum().item()

        feat, scimg = feat.to(device), scimg.to(device)

        optimizer.zero_grad()

        z_dist, output, im_out, mu, logvar = model(feat)

        masked_scimg = scimg * mask
        im_out_masked = im_out * mask

        imgs_edges = edge_loss_fn(masked_scimg)
        recon_edges = edge_loss_fn(im_out_masked)

        # edge_loss = F.mse_loss(recon_edges, imgs_edges)
        feat_rec_loss = criterion(output, feat)
        recon_loss = reconstruction_loss(masked_scimg, im_out_masked, distribution="gaussian")
        kld_loss, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        """
        # TODO REMOVE ONE OF BANDED
        mmd_loss_n_liver = torch.tensor(0.0).to(device)
        # Check for class samples and calculate MMD loss if present in the batch
        neutrophil_band_mask = (label == 7)
        if neutrophil_band_mask.any():
            z_neutrophil_band = z_dist[neutrophil_band_mask]
            z_neutrophil_band = z_neutrophil_band.to(device)
            mmd_loss_n_liver = mmd(z_neutrophil_band, ref_z_class_3)
        """

        mmd_loss_n_lung = torch.tensor(0.0).to(device)
        # Check for class samples and calculate MMD loss if present in the batch
        neutrophil_band_mask = (label == label_map['neutrophil_banded'])
        if neutrophil_band_mask.any():
            z_neutrophil_band = z_dist[neutrophil_band_mask]
            z_neutrophil_band = z_neutrophil_band.to(device)
            mmd_loss_n_lung = mmd(z_neutrophil_band, ref_z_class_n_lung)

        mmd_loss_n_blood = torch.tensor(0.0).to(device)
        # Check for class samples and calculate MMD loss if present in the batch
        neutrophil_segmented_mask = (label == label_map['neutrophil_segmented'])
        if neutrophil_segmented_mask.any():
            z_neutrophil_segmented = z_dist[neutrophil_segmented_mask]
            z_neutrophil_segmented = z_neutrophil_segmented.to(device)
            mmd_loss_n_blood = mmd(z_neutrophil_segmented, ref_z_class_n_blood)


        train_loss = ((cff_feat_rec * feat_rec_loss) + (cff_im_rec * recon_loss) + (cff_kld * kld_loss) + (cff_mmd_n_blood * mmd_loss_n_blood) + (cff_mmd_n_lung * mmd_loss_n_lung)) # (cff_mmd_m * mmd_loss_monocyte)

        train_loss.backward()
        optimizer.step()

        loss += train_loss.data.cpu()
        acc_featrec_loss += feat_rec_loss.data.cpu()
        acc_imrec_loss += recon_loss.data.cpu()
        kl_div_loss += kld_loss.data.cpu()
        mmd_loss_n_blood += mmd_loss_n_blood.data.cpu()
        # mmd_loss_n_liver += mmd_loss_n_liver.data.cpu() #TODO  USE IF NECESSARY
        mmd_loss_n_lung += mmd_loss_n_lung.data.cpu()

        if epoch % 10 == 0:
            all_means.append(mu.data.cpu().numpy())
            all_logvars.append(logvar.data.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_z.append(z_dist.data.cpu().numpy())
            all_z_n_band.append(z_neutrophil_band.data.cpu().numpy())
            # all_z_monocyte.append(z_mono.data.cpu().numpy())
            all_z_n_segmented.append(z_neutrophil_segmented.data.cpu().numpy())

        # y_true.extend(label.cpu().numpy())
        # _, predicted = torch.max(class_pred.data, 1)
        # y_pred.extend(predicted.cpu().numpy())

    loss = loss / len(train_dataloader)
    acc_featrec_loss = acc_featrec_loss / len(train_dataloader)
    acc_imrec_loss = acc_imrec_loss / len(train_dataloader)
    kl_div_loss = kl_div_loss / len(train_dataloader)
    mmd_loss_n_blood = mmd_loss_n_blood / len(train_dataloader)
    # mmd_loss_n_liver = mmd_loss_n_liver / len(train_dataloader) #TODO  USE IF NECESSARY
    mmd_loss_n_lung = mmd_loss_n_lung / len(train_dataloader)
    # f1 = f1_score(y_true, y_pred, average='weighted')


    print("epoch : {}/{}, loss = {:.6f}, feat_loss = {:.6f}, imrec_loss = {:.6f}, kl_div = {:.6f}, mmd_loss_n_blood = {:.6f}, mmd_loss_n_lung = {:.6f} ".format
          (epoch + 1, epochs, loss.item(), acc_featrec_loss.item(), acc_imrec_loss.item(), kl_div_loss.item(), mmd_loss_n_blood.item(), mmd_loss_n_lung.item()))

    with open(result_file, "a") as f:
        f.write(f"Epoch {epoch + 1}: Loss = {loss.item():.6f}, Feat_Loss = {acc_featrec_loss.item():.6f}, "
                f"Img_Rec_Loss = {acc_imrec_loss.item():.6f}, KL_DIV = {kl_div_loss.item():.6f}, "
                f"MMD_Loss_n = {mmd_loss_n_blood.item():.6f}, MMD_Loss_myle = {mmd_loss_n_lung.item():.6f} \n")

    if epoch % 10 == 0:
        # latent_values_per_epoch = [np.stack((m, lv), axis=-1) for m, lv in zip(all_means, all_logvars)]
        # latent_values = np.concatenate(latent_values_per_epoch, axis=0)

        print(f"Number of neutrophil banded instances: {count_neutrophil_banded}")
        print(f"Number of neutrophil segmented instances: {count_neutrophil_segmented}")

        latent_filename = os.path.join(latent_dir, f'latent_epoch_{epoch}.npy')
        concatenated_means = np.concatenate(all_means, axis=0)
        np.save(latent_filename, concatenated_means)
        print(f"Latent data is saved for epoch {epoch + 1}, Shape: {concatenated_means.shape}")

        label_filename = os.path.join(label_dir, f'label_epoch_{epoch+1}.npy')
        np.save(label_filename, np.array(all_labels))
        print(f"Label data is saved for epoch {epoch}")

        z_filename = os.path.join(z_dir, f'z_epoch_{epoch}.npy')
        np.save(z_filename, np.concatenate(all_z, axis=0))

        log_filename = os.path.join(log_dir, f'log_epoch_{epoch}.npy')
        np.save(log_filename, np.concatenate(all_logvars, axis=0))

        # CHECK FILE NAME NOT SAME
        neutrophil_band_z_filename = os.path.join(neutrophil_banded_z_dir, f'neutrophil_z_band_{epoch}.npy')
        concatenated_z_band = np.concatenate(all_z_n_band, axis=0)
        print(f"Shape of concatenated_z for neutrophil banded before saving: {concatenated_z_band.shape}")
        np.save(neutrophil_band_z_filename, concatenated_z_band)

        neutrophil_segment_z_filename = os.path.join(neutrophil_segment_z_dir, f'neutrophil_z_segment_{epoch}.npy')
        concatenated_z_seg = np.concatenate(all_z_n_segmented, axis=0)
        print(f"Shape of concatenated_z for neutrophil segmented before saving: {concatenated_z_seg.shape}")
        np.save(neutrophil_segment_z_filename, concatenated_z_seg)

        for i, img in enumerate(masked_scimg):
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            filename = f"{i}-{epoch}.jpg"
            filepath = os.path.join(save_img_dir, filename)
            cv2.imwrite(filepath, img_np * 255)

            mask_np = mask[i].cpu().numpy().squeeze()
            mask_filename = f"{i}-{epoch}_mask.jpg"
            mask_filepath = os.path.join(save_mask_dir, mask_filename)
            cv2.imwrite(mask_filepath, mask_np * 255)



    model.eval()

    if epoch % 10 == 0:
        latent_data = np.load(latent_filename)
        # latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
        print(latent_data.shape)
        all_labels_array = np.array(all_labels)
        # print("Labels array shape:", all_labels_array.shape)

        # Filter out the 'erythroblast' class
        erythroblast_class_index = label_map['erythroblast']
        mask = all_labels_array != erythroblast_class_index
        filtered_latent_data = latent_data[mask]
        filtered_labels = all_labels_array[mask]

        # UMAP for latent space
        latent_data_umap = UMAP(n_neighbors=13, min_dist=0.1, n_components=2, metric='euclidean').fit_transform(
            filtered_latent_data)

        fig = plt.figure(figsize=(12, 10), dpi=150)
        gs = GridSpec(1, 2, width_ratios=[4, 1], figure=fig)

        ax = fig.add_subplot(gs[0])
        scatter = ax.scatter(latent_data_umap[:, 0], latent_data_umap[:, 1], s=100, c=filtered_labels, cmap='Spectral')
        ax.set_aspect('equal')

        x_min, x_max = np.min(latent_data_umap[:, 0]), np.max(latent_data_umap[:, 0])
        y_min, y_max = np.min(latent_data_umap[:, 1]), np.max(latent_data_umap[:, 1])

        zoom_factor = 0.40
        padding_factor = 0.3

        x_range = (x_max - x_min) * zoom_factor
        y_range = (y_max - y_min) * zoom_factor

        center_x = (x_max + x_min) / 2
        center_y = (y_max + y_min) / 2

        new_x_min = center_x - (x_range * (1 + padding_factor))
        new_x_max = center_x + (x_range * (1 + padding_factor))
        new_y_min = center_y - (y_range * (1 + padding_factor))
        new_y_max = center_y + (y_range * (1 + padding_factor))

        ax.set_xlim(new_x_min, new_x_max)
        ax.set_ylim(new_y_min, new_y_max)

        ax.set_title(f'Latent Space Representation - (Epoch {epoch})', fontsize=18)
        ax.set_xlabel('UMAP Dimension 1', fontsize=16)
        ax.set_ylabel('UMAP Dimension 2', fontsize=16)

        ax_legend = fig.add_subplot(gs[1])
        ax_legend.axis('off')

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


        ref_z_class_n_lung_cpu = ref_z_class_n_lung.cpu().numpy() if ref_z_class_n_lung.is_cuda else ref_z_class_n_lung.numpy()

        neutrophil_band_z_data = np.load(neutrophil_band_z_filename)
        # latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
        print(f"Loaded neutrophil banded z data with shape: {neutrophil_band_z_data.shape}")


        # Proceed with UMAP visualization
        combined_data = np.vstack([neutrophil_band_z_data, ref_z_class_n_lung_cpu])
        umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
        umap_embedding = umap_reducer.fit_transform(combined_data)

        split_point = neutrophil_band_z_data.shape[0]
        umap_z_neutrophil_band = umap_embedding[:split_point, :]
        umap_ref_z_class_n_lung = umap_embedding[split_point:, :]

        plt.figure(figsize=(12, 6))
        plt.scatter(umap_z_neutrophil_band[:, 0], umap_z_neutrophil_band[:, 1], s=10, label='Model Neutrophil_Band')
        plt.scatter(umap_ref_z_class_n_lung[:, 0], umap_ref_z_class_n_lung[:, 1], s=10, label='Reference Neutrophil_Lung', alpha=0.6)
        plt.title('UMAP Visualization of Neutrophil Banded-Lung Tissue')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend()
        plt.grid(False)
        plt.savefig(os.path.join(umap_dir, f'umap_neutrophil_banded_comparison_{epoch}.png'))
        plt.close()

        ref_z_class_n_blood_cpu = ref_z_class_n_blood.cpu().numpy() if ref_z_class_n_blood.is_cuda else ref_z_class_n_blood.numpy()


        neutrophil_segment_z_data = np.load(neutrophil_segment_z_filename)
        # latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
        print(f"Loaded neutrophil segmented z data with shape: {neutrophil_segment_z_data.shape}")

        # Proceed with UMAP visualization
        combined_data = np.vstack([neutrophil_segment_z_data, ref_z_class_n_blood_cpu])
        umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
        umap_embedding = umap_reducer.fit_transform(combined_data)

        split_point = neutrophil_segment_z_data.shape[0]
        umap_z_neutrophil_segment = umap_embedding[:split_point, :]
        umap_ref_z_class_n_blood = umap_embedding[split_point:, :]

        plt.figure(figsize=(12, 6))
        plt.scatter(umap_z_neutrophil_segment[:, 0], umap_z_neutrophil_segment[:, 1], s=10, label='Model Neutrophil_Segment')
        plt.scatter(umap_ref_z_class_n_blood[:, 0], umap_ref_z_class_n_blood[:, 1], s=10, label='Reference Neutrophi_Blood',
                    alpha=0.6)
        plt.title('UMAP Visualization of Neutrophil Segmented-Blood Tissue')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend()
        plt.grid(False)
        plt.savefig(os.path.join(umap_dir, f'umap_neutrophil_segmented_comparison_{epoch}.png'))
        plt.close()

    neutrophil_banded_label = 7

    file_name = "reconstructed-neutrophil_banded_lung/"
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    # Process and visualize only for specified neutrophil classes
    for i in range(30):
        ft, img, mask, lbl, _ = train_dataset[i]

        # Check if the label is for neutrophil banded or segmented
        if lbl == neutrophil_banded_label:
            ft = np.expand_dims(ft, axis=0)
            ft = torch.tensor(ft, dtype=torch.float).to(device)  # Ensure correct dtype

            _, _, im_out, _, _ = model(ft)
            im_out = im_out.data.cpu().numpy().squeeze()
            im_out = np.moveaxis(im_out, 0, 2)
            img = np.moveaxis(img, 0, 2)
            im = np.concatenate([img, im_out], axis=1)

            if epoch % 10 == 0:
                cv2.imwrite(os.path.join(file_name, f"{i}-{epoch}.jpg"), im * 255)

    neutrophil_segmented_label = 8
    file_name = "reconstructed-neutrophil_segment_blood/"
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    # Process and visualize only for specified neutrophil classes
    for i in range(30):
        ft, img, mask, lbl, _ = train_dataset[i]

        # Check if the label is for neutrophil banded or segmented
        if lbl == neutrophil_segmented_label:
            ft = np.expand_dims(ft, axis=0)
            ft = torch.tensor(ft, dtype=torch.float).to(device)  # Ensure correct dtype

            _, _, im_out, _, _ = model(ft)
            im_out = im_out.data.cpu().numpy().squeeze()
            im_out = np.moveaxis(im_out, 0, 2)
            img = np.moveaxis(img, 0, 2)
            im = np.concatenate([img, im_out], axis=1)

            if epoch % 10 == 0:
                cv2.imwrite(os.path.join(file_name, f"{i}-{epoch}.jpg"), im * 255)


script_dir = os.path.dirname(__file__)

model_save_path = os.path.join(script_dir, 'trained_model4cp2_new5_std_gen_2_cp.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Trained model saved to {model_save_path}")

with open(result_file, "a") as f:
    f.write("Training completed.\n")
