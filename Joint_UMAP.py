import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from umap import UMAP
import torch.nn.functional as F
from sklearn.metrics import f1_score
from Dataloader_4 import Dataloader, label_map
from SSIM import SSIM
from model4 import VariationalAutoencodermodel4
from matplotlib.colors import ListedColormap
import os
from mmd import MMDLoss, RBF
import time
import torchvision
import cv2
from mmd_loss_2 import mmd
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
# import mrcnn.config
# import mrcnn.model_feat_extract
import numpy as np



#data for myeloid and neutrophils
#gene latent
latent_dir_ge = 'latent_variables_GE_3'
z_dir_ge = 'z_variables_GE_3'
epoch_of_gen = 290

class_labels_gen = [0, 1, 2, 3]
# monocyte : class 1
# myeloblast : class 2
# basophil : class 0
# neutrophil : class 3

class_label_n = 3
class_label_m = 1
class_label_myeloid = 2

z_filename_n = os.path.join(z_dir_ge, f'class_{class_label_n}_z_epoch_{epoch_of_gen}.npy')
z_filename_myeloid = os.path.join(z_dir_ge, f'class_{class_label_myeloid}_z_epoch_{epoch_of_gen}.npy')
ref_z_class_neutrophils = np.load(z_filename_n)
ref_z_class_myeloid = np.load(z_filename_myeloid)



epoch = 150

neutrophil_z_dir = 'z_data4cp2_new5_std_gen_2'
neutrophil_z_path = os.path.join(neutrophil_z_dir, f'neutrophil_z_eopch_{epoch}.npy')
neutrophil_z_data = np.load(neutrophil_z_path)
# latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
print(f"Loaded neutrophil z data with shape: {neutrophil_z_data.shape}")

myle_z_dir = 'z_data4cp2_new5_std_gen_2'
myeloblast_z_path = os.path.join(myle_z_dir, f'myle_z_eopch_{epoch}.npy')
myle_z_data = np.load(myeloblast_z_path)
print(f"Loaded myeloblast z data with shape: {myle_z_data.shape}")

umap_dir = 'joint_latent_m_neutrophils'
if not os.path.exists(umap_dir):
    os.makedirs(umap_dir)


# UMAP1

combined_data = np.vstack([neutrophil_z_data, ref_z_class_neutrophils])
umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
umap_embedding = umap_reducer.fit_transform(combined_data)

split_point = neutrophil_z_data.shape[0]
umap_z_neutrophil = umap_embedding[:split_point, :]
umap_ref_z_class_3 = umap_embedding[split_point:, :]

plt.figure(figsize=(12, 6))
plt.scatter(umap_z_neutrophil[:, 0], umap_z_neutrophil[:, 1], s=10, label='Model Neutrophil')
plt.scatter(umap_ref_z_class_3[:, 0], umap_ref_z_class_3[:, 1], s=10, label='Reference Neutrophil', alpha=0.6)
plt.title('UMAP Visualization of Neutrophil Latent Representations (Post-Training)')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend()
plt.grid(False)
plt.savefig(os.path.join(umap_dir, f'umap_neutrophil_comparison_{epoch}.png'))
plt.close()

# UMAP2

combined_data = np.vstack([myle_z_data, ref_z_class_myeloid])
umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
umap_embedding = umap_reducer.fit_transform(combined_data)

split_point = myle_z_data.shape[0]
umap_z_myle = umap_embedding[:split_point, :]
umap_ref_z_class_2 = umap_embedding[split_point:, :]

plt.figure(figsize=(12, 6))
plt.scatter(umap_z_myle[:, 0], umap_z_myle[:, 1], s=10, label='Model Myeloblast')
plt.scatter(umap_ref_z_class_2[:, 0], umap_ref_z_class_2[:, 1], s=10, label='Reference Myeloblast', alpha=0.6)
plt.title('UMAP Visualization of Myeloblast Latent Representations (Post-Training)')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend()
plt.grid(False)
plt.savefig(os.path.join(umap_dir, f'umap_myeloblast_comparison_{epoch}.png'))
plt.close()

#data for monocyte
#gene latent
z_filename_m = os.path.join(z_dir_ge, f'class_{class_label_m}_z_epoch_{epoch_of_gen}.npy')
ref_z_class_mono = np.load(z_filename_m)

#image latent
epoch = 150
monocyte_z_dir = 'z_data4cp2_new5_std_gen_2_mono'
mono_z_path = os.path.join(monocyte_z_dir, f'monocyte_z_eopch_{epoch}.npy')
monocyte_z_data = np.load(mono_z_path)



# UMAP 3
combined_data = np.vstack([monocyte_z_data, ref_z_class_mono])
umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
umap_embedding = umap_reducer.fit_transform(combined_data)

split_point = monocyte_z_data.shape[0]
umap_z_mono = umap_embedding[:split_point, :]
umap_ref_z_class_1 = umap_embedding[split_point:, :]

plt.figure(figsize=(12, 6))
plt.scatter(umap_z_mono[:, 0], umap_z_mono[:, 1], s=10, label='Model monocyte')
plt.scatter(umap_ref_z_class_1[:, 0], umap_ref_z_class_1[:, 1], s=10, label='Reference Monocyte', alpha=0.6)
plt.title('UMAP Visualization of Monocyte Latent Representations (Post-Training)')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend()
plt.grid(False)
plt.savefig(os.path.join(umap_dir, f'umap_monocyte_comparison_{epoch}_training.png'))
plt.close()


# data for n.baded and n.segmented
# gene data
latent_dir_ge_2 = 'latent_variables_GE_3_cp'
z_dir_ge_2 = 'z_variables_GE_3_cp'

class_label_n_blood = 2
class_label_n_lung = 5

z_filename_n_blood = os.path.join(z_dir_ge_2, f'class_{class_label_n_blood}_z_epoch_{epoch_of_gen}.npy')
ref_z_class_n_blood = np.load(z_filename_n_blood)

z_filename_n_lung = os.path.join(z_dir_ge_2, f'class_{class_label_n_lung}_z_epoch_{epoch_of_gen}.npy')
ref_z_class_n_lung = np.load(z_filename_n_lung)

# image data
epoch_2 = 280

n_banded_lung_z_dir = 'banded_z_data4cp2_new5_std_gen_2_cp'
n_banded_lung_z_path = os.path.join(n_banded_lung_z_dir, f'neutrophil_z_band_{epoch}.npy')
neutrophil_band_z_data = np.load(n_banded_lung_z_path)

n_segment_blood_z_dir = 'segment_z_data4cp2_new5_std_gen_2_cp'
n_segment_blood_z_path = os.path.join(n_segment_blood_z_dir, f'neutrophil_z_segment_{epoch}.npy')
neutrophil_segment_z_data = np.load(n_segment_blood_z_path)


# UMAP 4
combined_data = np.vstack([neutrophil_band_z_data, ref_z_class_n_lung])
umap_reducer = UMAP(n_neighbors=13, min_dist=1, n_components=2, metric='euclidean')
umap_embedding = umap_reducer.fit_transform(combined_data)

split_point = neutrophil_band_z_data.shape[0]
umap_z_neutrophil_band = umap_embedding[:split_point, :]
umap_ref_z_class_n_lung = umap_embedding[split_point:, :]

plt.figure(figsize=(12, 6))
plt.scatter(umap_z_neutrophil_band[:, 0], umap_z_neutrophil_band[:, 1], s=10, label='Model Neutrophil_Band')
plt.scatter(umap_ref_z_class_n_lung[:, 0], umap_ref_z_class_n_lung[:, 1], s=10, label='Reference Neutrophil_Lung',
            alpha=0.6)
plt.title('UMAP Visualization of Neutrophil Banded-Lung Tissue')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend()
plt.grid(False)
plt.savefig(os.path.join(umap_dir, f'umap_neutrophil_banded_comparison_{epoch}.png'))
plt.close()

# UMAP 5
combined_data = np.vstack([neutrophil_segment_z_data, ref_z_class_n_blood])
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

# COMBINED UMAP
combined_all_data = np.vstack([
    neutrophil_z_data,  # Loaded earlier
    ref_z_class_neutrophils,
    myle_z_data,  # Loaded earlier
    ref_z_class_myeloid,
    monocyte_z_data,  # Loaded earlier
    ref_z_class_mono,
    neutrophil_band_z_data,  # Loaded earlier
    ref_z_class_n_lung,
    neutrophil_segment_z_data,
    ref_z_class_n_blood
])

labels = np.concatenate([
    np.repeat("Model Neutrophil", neutrophil_z_data.shape[0]),
    np.repeat("Reference Neutrophil", ref_z_class_neutrophils.shape[0]),
    np.repeat("Model Myeloblast", myle_z_data.shape[0]),
    np.repeat("Reference Myeloblast", ref_z_class_myeloid.shape[0]),
    np.repeat("Model Monocyte", monocyte_z_data.shape[0]),
    np.repeat("Reference Monocyte", ref_z_class_mono.shape[0]),
    np.repeat("Model Neutrophil Band", neutrophil_band_z_data.shape[0]),
    np.repeat("Reference Neutrophil Lung", ref_z_class_n_lung.shape[0]),
    np.repeat("Model Neutrophil Segment", neutrophil_segment_z_data.shape[0]),
    np.repeat("Reference Neutrophil Blood", ref_z_class_n_blood.shape[0])
])

umap_reducer_all = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
umap_embedding_all = umap_reducer_all.fit_transform(combined_all_data)

plt.figure(figsize=(12, 8))
unique_labels = np.unique(labels)
for label in unique_labels:
    idx = labels == label
    plt.scatter(umap_embedding_all[idx, 0], umap_embedding_all[idx, 1], s=10, label=label, alpha=0.6)

plt.title('UMAP Visualization of Combined Latent Representations')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend()
plt.grid(False)
plt.savefig(os.path.join(umap_dir, f'umap_all_{epoch}.svg'))
plt.close()












