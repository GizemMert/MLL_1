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
adata = anndata.read_h5ad('s_data_tabula.h5ad')
X = adata.X
print("Maximum value in X:", X.max())
print("Minimum value in X:", X.min())

label_mapping = {label: index for index, label in enumerate(adata.obs['cell_ontology_class'].cat.categories)}
numeric_labels = adata.obs['cell_ontology_class'].map(label_mapping).to_numpy()
inverse_label_map = {v: k for k, v in label_mapping.items()}

batch_size = 128
epochs = 160
beta = 0.2
cff_rec = 0.4
cff_emd = 0.4

from torch.utils.data import Dataset


class GeneExpressionDataset(Dataset):
    def __init__(self, expressions, labels, scvi_embeddings):
        self.expressions = expressions
        self.labels = labels
        self.scvi_embeddings = scvi_embeddings

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, idx):
        expression = self.expressions[idx]
        label = self.labels[idx]
        scvi_embedding = self.scvi_embeddings[idx]

        expression = expression.view(1, -1)

        return expression, label, scvi_embedding




X_dense = X.toarray()  # Convert sparse matrix to dense
X_tensor = torch.tensor(X_dense, dtype=torch.float32)
label_tensor = torch.tensor(numeric_labels, dtype=torch.long)
scvi_tensor = torch.tensor(adata.obsm["X_scvi"], dtype=torch.float32)

torch.manual_seed(42)
# Initialize dataset
dataset = GeneExpressionDataset(X_tensor, label_tensor, scvi_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=1)
print("loading done")
heatmap_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_shape = X.shape[1]  # number of genes
model = VAE_GE(latent_dim=50).to(device)


optimizer = Adam(model.parameters(), lr=0.01)


latent_dir = 'latent_variables_GE_plus'
if not os.path.exists(latent_dir):
    os.makedirs(latent_dir)

z_dir = 'z_variables_GE_plus'
if not os.path.exists(z_dir):
    os.makedirs(z_dir)

umap_dir = 'umap_GE_plus'
if not os.path.exists(umap_dir):
    os.makedirs(umap_dir)

result_dir = "training_results_GE_plus"
os.makedirs(result_dir, exist_ok=True)
result_file = os.path.join(result_dir, "training_results_GE_plus.txt")

def kl_loss(mu, logvar):
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # kl_loss /= batch_size * input_shape
    return kl_loss

def rec_loss(recgen, gen):
    recon_loss = F.mse_loss(recgen, gen, reduction='mean')
    return recon_loss


def embedding_loss(z, scvi_embedding):
    loss = F.mse_loss(z, scvi_embedding, reduction='mean')
    return loss
# Training loop

for epoch in range(epochs):
    loss = 0.0
    embedd_loss = 0.0
    acc_recgen_loss = 0.0
    acc_kl_loss = 0.0

    model.train()

    if epoch % 10 == 0:
        all_means = []
        all_labels = []
        all_z = []


    for gen, label, scvi_embedding in dataloader:
        gen = gen.to(device)
        label = label.to(device)
        scvi_embedding = scvi_embedding.to(device)
        # print("scvi shape:", scvi_embedding.shape)
        optimizer.zero_grad()

        # Forward pass
        z, recgen, mu, logvar = model(gen)

        # print("z shape: ", z.shape)
        recon_loss = rec_loss(recgen, gen)
        kl_div_loss = kl_loss(mu, logvar)
        scvi_embedding_loss = embedding_loss(z, scvi_embedding)
        train_loss = (cff_rec*recon_loss) + (beta*kl_div_loss) + (cff_emd*scvi_embedding_loss)


        # Backward pass
        train_loss.backward()
        optimizer.step()

        loss +=train_loss.data.cpu()
        acc_recgen_loss +=recon_loss.data.cpu()
        acc_kl_loss +=kl_div_loss.data.cpu()
        embedd_loss +=scvi_embedding_loss.data.cpu()
        if epoch % 10 == 0:
            all_means.append(mu.detach().cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_z.append(z.detach().cpu().numpy())

    loss = loss / len(dataloader)
    acc_recgen_loss = acc_recgen_loss / len(dataloader)
    acc_kl_loss = acc_kl_loss / len(dataloader)
    emb_loss =embedd_loss / len(dataloader)

    print("epoch : {}/{}, loss = {:.6f}, rec_loss = {:.6f}, kl_div = {:.6f}, embed_loss = {:.6f}".format
          (epoch + 1, epochs, loss.item(), acc_recgen_loss.item(), acc_kl_loss.item(), emb_loss.item()))

    with open(result_file, "a") as f:
        f.write(f"Epoch {epoch + 1}: Loss = {loss.item():.6f}, rec_Loss = {acc_recgen_loss.item():.6f} "
                f"KL_Loss = {acc_kl_loss.item():.6f}, Embedd_Loss = {emb_loss.item():.6f} \n")

    if epoch % 10 == 0:
        latent_filename = os.path.join(latent_dir, f'latent_epoch_{epoch}.npy')
        np.save(latent_filename, np.concatenate(all_means, axis=0))

        z_filename = os.path. join(z_dir, f'z_epoch_{epoch}.npy')
        np.save(z_filename, np.concatenate(all_z, axis=0))

    model.eval()

    if epoch % 10 == 0:
        # Load all latent representations
        latent_data = np.load(latent_filename)
        latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
        print(latent_data_reshaped.shape)
        all_labels_array = np.array(all_labels)
        print("Labels array shape:", all_labels_array.shape)

        latent_data_umap = UMAP(n_neighbors=13, min_dist=0.1, n_components=2, metric='euclidean').fit_transform(
            latent_data_reshaped)


        plt.figure(figsize=(12, 10), dpi=150)
        scatter = plt.scatter(latent_data_umap[:, 0], latent_data_umap[:, 1], s=1, c=all_labels_array, cmap='plasma')

        color_map = plt.cm.plasma(np.linspace(0, 1, len(set(all_labels_array))))
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

model.eval()
file_name = "heat_map/"
accumulated_mae = np.zeros(input_shape)
total_samples = 0

if not os.path.exists(file_name):
    os.makedirs(file_name)

for gen, _ , _ in dataloader:
    gen = gen.to(device)
    with torch.no_grad():
        _, recgen, _, _ = model(gen)


    recgen = recgen.detach().cpu().numpy()
    gen = gen.detach().cpu().numpy()

    abs_errors = np.abs(gen - recgen)
    accumulated_mae += abs_errors.sum(axis=0)
    total_samples += gen.shape[0]

average_mae = accumulated_mae / total_samples

# Plotting the averaged 1D heatmap for all samples
plt.figure(figsize=(50, 5))
heatmap_data = average_mae[np.newaxis, :]
plt.imshow(heatmap_data, cmap='hot', aspect='auto')
plt.colorbar(label='MAE')
plt.xlabel('Features')
plt.xticks(np.arange(0, len(average_mae), step=max(len(average_mae) // 10, 1)),
           rotation=90)
plt.yticks([])
plt.title('Average MAE Across All Samples')

plt.savefig(os.path.join(file_name, f"heatmap_all_sample_plus.jpg"))
print("it is saved ")
plt.close()



script_dir = os.path.dirname(__file__)

model_save_path = os.path.join(script_dir, 'trained_model_GE_plus.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Trained model saved to {model_save_path}")

with open(result_file, "a") as f:
    f.write("Training completed.\n")