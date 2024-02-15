import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import anndata
from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
import os
from Model_Vae_GE_2 import VAE_GE
import scanpy as sc

# Load data
adata = anndata.read_h5ad('sdata_d.h5ad')
sc.pp.combat(adata, key='donor')
X = adata.layers["scran_normalization"]  # normalized gene expression matrix


label_mapping = {label: index for index, label in enumerate(adata.obs['cell_ontology_class'].cat.categories)}
numeric_labels = adata.obs['cell_ontology_class'].map(label_mapping).to_numpy()
inverse_label_map = {v: k for k, v in label_mapping.items()}

batch_size = 128
epochs = 150
beta = 4
# Create dataset and dataloader
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


optimizer = Adam(model.parameters(), lr=0.001)


latent_dir = 'latent_variables_GE_1'
if not os.path.exists(latent_dir):
    os.makedirs(latent_dir)
umap_dir = 'umap_GE_1'
if not os.path.exists(umap_dir):
    os.makedirs(umap_dir)

result_dir = "training_results_GE_1"
os.makedirs(result_dir, exist_ok=True)
result_file = os.path.join(result_dir, "training_results_GE_1.txt")

def kl_loss(mu, logvar):
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss /= batch_size * input_shape
    return kl_loss

def rec_loss(recgen, gen):
    recon_loss = F.mse_loss(recgen, gen, reduction='mean')
    return recon_loss

# Training loop

for epoch in range(epochs):
    loss = 0
    acc_recgen_loss  =0
    acc_kl_loss = 0

    model.train()

    if epoch % 10 == 0:
        all_means = []
        all_labels = []

    for gen, label in dataloader:
        gen = gen.float()
        label = label.long().to(device)
        gen = gen.to(device)
        optimizer.zero_grad()

        # Forward pass
        z, recgen, mu, logvar = model(gen)

        recon_loss = rec_loss(recgen, gen)
        kl_div_loss = kl_loss(mu, logvar)
        train_loss = recon_loss + (beta*kl_div_loss)

        # Backward pass
        train_loss.backward()
        optimizer.step()

        loss +=train_loss.data.cpu()
        acc_recgen_loss +=recon_loss.data.cpu()
        acc_kl_loss +=kl_div_loss.data.cpu()

        if epoch % 10 == 0:
            all_means.append(mu.detach().cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    loss = loss / len(dataloader)
    acc_recgen_loss = acc_recgen_loss / len(dataloader)
    acc_kl_loss = acc_kl_loss / len(dataloader)

    print("epoch : {}/{}, loss = {:.6f}, rec_loss = {:.6f}, kl_div = {:.6f}".format
          (epoch + 1, epochs, loss.item(), acc_recgen_loss.item(), acc_kl_loss.item()))

    with open(result_file, "a") as f:
        f.write(f"Epoch {epoch + 1}: Loss = {loss.item():.6f}, rec_Loss = {acc_recgen_loss.item():.6f}, "
                f"KL_Loss = {acc_kl_loss.item():.6f} \n")

    if epoch % 10 == 0:
        latent_filename = os.path.join(latent_dir, f'latent_epoch_{epoch}.npy')
        np.save(latent_filename, np.concatenate(all_means, axis=0))


model.eval()
file_name = "heat_map_1/"

if not os.path.exists(file_name):
    os.makedirs(file_name)

for sample_index, (gen, label) in enumerate(heatmap_dataloader):
    if sample_index >= 30:
        break

    gen = gen.to(device)
    _, recgen, _, _ = model(gen)

    recgen = recgen.detach().cpu().numpy()
    gen = gen.detach().cpu().numpy()

    mae_per_feature = np.abs(gen.squeeze() - recgen.squeeze())

    # Plotting the 1D heatmap for this sample
    plt.figure(figsize=(50, 3))
    heatmap_data = mae_per_feature[np.newaxis, :]
    plt.imshow(heatmap_data, cmap='hot', aspect='auto')
    plt.colorbar(label='MAE')
    plt.xlabel('Features')
    plt.xticks(range(len(mae_per_feature)), rotation=90)
    plt.yticks([])
    plt.title(f'MAE for Sample {sample_index + 1}')

    plt.savefig(os.path.join(file_name, f"heatmap-sample-{sample_index + 1}.jpg"))
    plt.close()






script_dir = os.path.dirname(__file__)

model_save_path = os.path.join(script_dir, 'trained_model_GE_1.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Trained model saved to {model_save_path}")

with open(result_file, "a") as f:
    f.write("Training completed.\n")