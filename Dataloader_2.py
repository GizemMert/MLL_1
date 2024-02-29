import gzip
import numpy as np
import pickle
import os
import cv2
import torch
from torch.utils.data import Dataset
import random


equivalent_classes = {

    #  INT-20 dataset
    '01-NORMO': 'erythroblast',
    '04-LGL': "unknown",  # atypical
    '05-MONO': 'monocyte',
    '08-LYMPH-neo': 'lymphocyte_atypical',
    '09-BASO': 'basophil',
    '10-EOS': 'eosinophil',
    '11-STAB': 'neutrophil_banded',
    '12-LYMPH-reaktiv': 'lymphocyte_atypical',
    '13-MYBL': 'myeloblast',
    '14-LYMPH-typ': 'lymphocyte_typical',
    '15-SEG': 'neutrophil_segmented',
    '16-PLZ': "unknown",
    '17-Kernschatten': 'smudge_cell',
    '18-PMYEL': 'promyelocyte',
    '19-MYEL': 'myelocyte',
    '20-Meta': 'metamyelocyte',
    '21-Haarzelle': "unknown",
    '22-Atyp-PMYEL': "unknown",
}

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


class Dataloader(Dataset):
    def __init__(self, split='train'):
        self.split = split

        features_mll_path = (
            "/lustre/groups/aih/raheleh.salehi/Master-thesis/Aug_features_datasets/Augmented-MLL-AML_MLLdataset.dat.gz")

        samples = {}
        with gzip.open(os.path.join(features_mll_path), "rb") as f:
            data = pickle.load(f)
            for d in data:
                data[d]["dataset"] = "MLL-AML"
                if "label" not in data[d].keys():
                    data[d]["label"] = d.split("_")[0]
            samples = {**samples, **data}
        print("[done]")

        samples2 = samples.copy()
        for s in samples2:
            if equivalent_classes[samples[s]["label"]] == "unknown":
                samples.pop(s, None)

        # loading images
        images = {}
        images_path = "/lustre/groups/aih/raheleh.salehi/Master-thesis/save_files/mll_images.pkl.gz"

        with gzip.open(os.path.join(images_path), "rb") as f:
            file_images = pickle.load(f)
        images = {**images, **file_images}
        print("[done]")

        self.samples = samples
        self.images = images

        data_keys = list(set(samples.keys()) & set(self.images.keys()))
        random.shuffle(data_keys)
        self.data = list(set(self.samples.keys()) & set(self.images.keys()))

        print("Total number of samples:", len(self.data))

    def __len__(self):
        if self.split == 'train':
            return int(len(self.data) * 1)  # 90% for training
        elif self.split == 'test':
            return len(self.data) - int(len(self.data) * 1)  # 10% for testing

    def get_samples_by_class(self, class_labels, n_samples=5):


        class_samples = {label: [] for label in class_labels}
        for key in self.data:
            label_fold = self.samples[key]['label']
            label_fold = equivalent_classes.get(label_fold, label_fold)
            label_fold = label_map.get(label_fold, -1)
            if label_fold in class_labels:
                img = self.images[key]
                class_samples[label_fold].append((img, label_fold))


        for label in class_samples:
            class_samples[label] = random.sample(class_samples[label], min(len(class_samples[label]), n_samples))

        return class_samples

    def save_class_samples(self, class_samples, base_save_dir):
        if not os.path.exists(base_save_dir):
            os.makedirs(base_save_dir)

        class_folders = {
            3: 'myeloblast',
            7: 'neutrophil_banded',
            8: 'neutrophil_segmented'
        }

        for label, samples in class_samples.items():
            class_dir = os.path.join(base_save_dir, class_folders.get(label, f"class_{label}"))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)


            for i, (img, _) in enumerate(samples):
                file_path = os.path.join(class_dir, f"sample_{i}.png")
                cv2.imwrite(file_path, img * 255)

train_dataset = Dataloader(split='train')

class_labels = [3, 7, 8]  # myeloblast, neutrophil banded, neutrophil segmented

class_samples = train_dataset.get_samples_by_class(class_labels)

save_dir = 'save_class_imaged'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


train_dataset.save_class_samples(class_samples, base_save_dir)
