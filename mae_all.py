import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import anndata
from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
import os
from Model_Vae_GE import VAE_GE

# Load data
adata = anndata.read_h5ad('sdata_d.h5ad')
X = adata.layers["scran_normalization"]  # normalized gene expression matrix
print("Maximum value in X:", X.max())
print("Minimum value in X:", X.min())


label_mapping = {label: index for index, label in enumerate(adata.obs['cell_ontology_class'].cat.categories)}
numeric_labels = adata.obs['cell_ontology_class'].map(label_mapping).to_numpy()
inverse_label_map = {v: k for k, v in label_mapping.items()}

batch_size = 128
epochs = 150
beta = 4


from torch.utils.data import Dataset


class GeneExpressionDataset(Dataset):
    def __init__(self, expressions, labels):
        self.expressions = expressions
        self.labels = labels
        print("done")

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, idx):
        expression = self.expressions[idx]
        label = self.labels[idx]
        return expression, label


# Convert to PyTorch tensors
X_dense = X.toarray()  # Convert sparse matrix to dense
X_tensor = torch.tensor(X_dense, dtype=torch.float32)
label_tensor = torch.tensor(numeric_labels, dtype=torch.long)

torch.manual_seed(42)
# Initialize dataset
dataset = GeneExpressionDataset(X_tensor, label_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=1)
print("loading done")
heatmap_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_shape = X.shape[1]  # number of genes
model = VAE_GE(input_shape=input_shape, latent_dim=30).to(device)
model_save_path = 'trained_model_GE.pth'
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()

file_name = "heat_map/"
accumulated_mae = np.zeros(input_shape)
total_samples = 0

if not os.path.exists(file_name):
    os.makedirs(file_name)

for gen, _ in dataloader:  # label is not needed for MAE calculation
    gen = gen.to(device)
    with torch.no_grad():  # No gradients needed for inference
        _, recgen, _, _ = model(gen)


    recgen = recgen.detach().cpu().numpy()
    gen = gen.detach().cpu().numpy()

    abs_errors = np.abs(gen - recgen)
    accumulated_mae += abs_errors.sum(axis=0)  # Sum across the batch, not mean, to accumulate errors
    total_samples += gen.shape[0]

average_mae = accumulated_mae / total_samples

# Plotting the averaged 1D heatmap for all samples
plt.figure(figsize=(50, 5))
heatmap_data = average_mae[np.newaxis, :]
im = plt.imshow(heatmap_data, cmap='hot', aspect='auto')
cbar = plt.colorbar(im, label='MAE', fraction=2, pad=0.04)
plt.xlabel('Features')
plt.xticks(np.arange(0, len(average_mae), step=max(len(average_mae) // 10, 1)),
           rotation=90)
plt.yticks([])
plt.title('Average MAE Across All Samples')

plt.savefig(os.path.join(file_name, f"heatmap_all_sample.jpg"))
print("it is saved ")
plt.close()