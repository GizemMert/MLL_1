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

inverse_label_map = {v: k for k, v in label_map.items()}  # inverse mapping for UMAP
epochs = 160
batch_size = 128
ngpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(label_map)
model = VariationalAutoencodermodel4(latent_dim=50)
model_name = 'AE-CFE-'

# Load the dataset
train_dataset = Dataloader(split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

class_labels = {
    'neutrophil_banded': 7,
    'neutrophil_segmented': 8,
    'myeloblast': 3,
    'promyelocyte': 4,
    'myelocyte': 5,
    'metamyelocyte': 6,
    'monocyte': 9,
    'basophil': 0
}

for class_name in class_labels.keys():
    dir_path = f"reconstructed-{class_name}/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_images_for_classes(dataloader, class_labels, n_images=5):
    saved_images_count = {label: 0 for label in class_labels.values()}  # Initialize count for each class

    for i, (feat, img, mask, lbl, _) in enumerate(dataloader.dataset):
        if lbl in saved_images_count and saved_images_count[lbl] < n_images:
            class_name = [name for name, label in class_labels.items() if label == lbl][0]  # Get class name from label
            save_path = os.path.join(f"reconstructed-{class_name}/", f"{saved_images_count[lbl]}.jpg")

            image_to_save = img.transpose(1, 2, 0)
            image_to_save = (image_to_save * 255).astype(np.uint8)


            if image_to_save.shape[2] == 1:
                image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_GRAY2BGR)
            else:
                image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)

            cv2.imwrite(save_path, image_to_save)
            saved_images_count[lbl] += 1

            if all(count >= n_images for count in saved_images_count.values()):
                break


save_images_for_classes(train_dataloader, class_labels)