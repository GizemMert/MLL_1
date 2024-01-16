import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Dataloader import Dataloader, label_map



epochs = 300
batch_size = 128

# Load the dataset
train_dataset = Dataloader(split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

for feat, scimg, mask, label, _ in train_dataloader:
    print(f"Size of mask: {mask.size()}")
    break  # Break after checking the first batch

