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

# Initialize dataset
dataset = GeneExpressionDataset(X_tensor, label_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=1)
print("loading done")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_shape = X.shape[1]  # number of genes
model = VAE_GE(input_shape=input_shape, latent_dim=30).to(device)


optimizer = Adam(model.parameters(), lr=0.001)


latent_dir = 'latent_variables_GE'
if not os.path.exists(latent_dir):
    os.makedirs(latent_dir)
umap_dir = 'umap_GE'
if not os.path.exists(umap_dir):
    os.makedirs(umap_dir)

result_dir = "training_results_GE"
os.makedirs(result_dir, exist_ok=True)
result_file = os.path.join(result_dir, "training_results_GE.txt")

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

        mae_values = []
        file_name = "heat_map/"

        if not os.path.exists(file_name):
            os.makedirs(file_name)
        for i in range(50):
            for i, (gen, label) in enumerate(dataloader):
                gen = gen.float().to(device)

            _, recgen, _, _ = model(gen)
            recgen = recgen.data.cpu().numpy()
            gen = gen.cpu().numpy()
            mae = np.mean(np.abs(gen - recgen))
            mae_values.append(mae)

            if epoch % 10 == 0:
                plt.figure(figsize=(20, 5))
                plt.imshow([mae_values], cmap='hot', aspect='auto')
                plt.colorbar(label='MAE')
                plt.xlabel('Samples')
                plt.ylabel('MAE')
                plt.title(f'Mean Absolute Error (MAE) Heatmap at Epoch {epoch}')

                plt.savefig(os.path.join(file_name, f"heatmap-epoch-{epoch}.jpg"))
                plt.close()



script_dir = os.path.dirname(__file__)

model_save_path = os.path.join(script_dir, 'trained_model_GE.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Trained model saved to {model_save_path}")

with open(result_file, "a") as f:
    f.write("Training completed.\n")