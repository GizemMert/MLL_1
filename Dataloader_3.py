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

    def __len__(self):
        if self.split == 'train':
            return int(len(self.data) * 1)  # 90% for training
        elif self.split == 'test':
            return len(self.data) - int(len(self.data) * 1)  # 10% for testing

    def __getitem__(self, index):
        if self.split == 'train':
            index = index % int(len(self.data) * 1)
        elif self.split == 'test':
            index = int(len(self.data) * 1) + index

        key = self.data[index]
        label_fold = self.samples[key]['label']
        label_fold = equivalent_classes.get(label_fold, label_fold)
        label_fold = label_map.get(label_fold, -1)
        img = self.images[key]
        mask = self.samples[key]['masks']
        if len(mask.shape) == 2:
            mask = mask[..., np.newaxis]
        bounding_box = self.samples[key]['rois']
        if len(bounding_box) == 1:
            bounding_box = bounding_box[0]
        w, h, _ = img.shape
        wm, hm, _ = mask.shape
        bounding_box = bounding_box / 400
        x0 = bounding_box[0] * w
        y0 = bounding_box[1] * h
        x1 = bounding_box[2] * w
        y1 = bounding_box[3] * h

        x2 = bounding_box[0] * wm
        y2 = bounding_box[1] * hm
        x3 = bounding_box[2] * wm
        y3 = bounding_box[3] * hm

        roi_cropped = img[
            max(0, int(y0) - 10):min(h, int(y1) + 20),
            max(0, int(x0) - 10):min(w, int(x1) + 20)
        ]
        roi_cropped = cv2.resize(roi_cropped, (128, 128))
        roi_cropped = roi_cropped / 255.
        roi_cropped = np.rollaxis(roi_cropped, 2, 0)

        mask_cropped = mask[
            max(0, int(y2) - 10):min(hm, int(y3) + 20),  # Crop vertically (rows)
            max(0, int(x2) - 10):min(wm, int(x3) + 20)   # Crop horizontally (columns)
        ]
        mask_dilation = mask_cropped.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask_dilation = cv2.dilate(mask_dilation, kernel, iterations=8)
        mask_dilation = mask_dilation.astype(np.float32)
        mask_dilation = cv2.resize(mask_dilation, (128, 128))
        mask_dilation = cv2.GaussianBlur(mask_dilation, (11, 11), sigmaX=0)

        if len(mask_dilation.shape) == 2:
            mask_dilation = mask_dilation[..., np.newaxis]

        mask_dilation = np.rollaxis(mask_dilation, 2, 0)

        feat = self.samples[key]['feats']
        feat = 2. * (feat - np.min(feat)) / np.ptp(feat) - 1
        feat = np.squeeze(feat)
        feat = np.rollaxis(feat, 2, 0)

        return feat, roi_cropped, mask_dilation, label_fold, key


"""
    def get_all_labels(self):
        all_labels = set()

        for key, sample in self.samples.items():
            label = sample.get('label', key.split("_")[0])
            all_labels.add(label)

        return list(all_labels)
"""