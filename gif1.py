import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from PIL import Image
from model4 import VariationalAutoencodermodel4, reparametrize
from Dataloader_4 import Dataloader
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import torch
import numpy as np
import cv2
import umap
import matplotlib.pyplot as plt
from Model_Vae_GE_2 import VAE_GE
from umap import UMAP
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# dimension = 30
# complex_manifold = cm.ComplexManifold(dimension)


epoch = 150
latent_dim = 30

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_1 = VariationalAutoencodermodel4(latent_dim=50)
model_save_path = 'trained_model4cp2_new5_std_gen_2.pth'
model_1.load_state_dict(torch.load(model_save_path, map_location=device))
model_1.to(device)
model_1.eval()

model_2 = VAE_GE(input_shape=2432, latent_dim=50).to(device)
model_save_path_2 = 'trained_model_GE_3.pth'
model_2.load_state_dict(torch.load(model_save_path_2, map_location=device))
model_2.to(device)
model_2.eval()

umap_dir = 'umap_trajectory_myeloblast_nsegmented_tresh_05'
if not os.path.exists(umap_dir):
    os.makedirs(umap_dir)

# Load all latent representations
latent_dir = 'latent_data4cp2_new5_std_gen_2'
latents_path = os.path.join(latent_dir, f'latent_epoch_{epoch}.npy')
label_dir = 'label_data4cp2_new5_std_gen_2'
labels_path = os.path.join(label_dir, f'label_epoch_151.npy')
neutrophil_z_dir = 'z_data4cp2_new5_std_gen_2'
neutrophil_z_path = os.path.join(neutrophil_z_dir, f'neutrophil_z_eopch_{epoch}.npy')
myeloblast_z_dir = 'z_data4cp2_new5_std_gen_2'
myeloblast_z_path = os.path.join(neutrophil_z_dir, f'myle_z_eopch_{epoch}.npy')

# Load all latent representations
latent_data = np.load(latents_path)
# latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
print("Latent data shape:", latent_data.shape)

# Load all neutrophil latent representations
neutrophil_data = np.load(neutrophil_z_path)
# latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
print("Latent data shape:", latent_data.shape)

neutrophil_data = np.load(myeloblast_z_path)
# latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
print("Latent data shape:", latent_data.shape)

# Load all labels
all_labels_array = np.load(labels_path)
print("Labels array shape:", all_labels_array.shape)

# print("Labels array shape:", all_labels_array.shape)

# Filter out the 'erythroblast' class
erythroblast_class_index = label_map['erythroblast']
neutrophil_banded_index = label_map['neutrophil_banded']
segmented_index = label_map['neutrophil_segmented']
mask = all_labels_array != erythroblast_class_index
mask2 = (all_labels_array == neutrophil_banded_index) | (all_labels_array == segmented_index)
filtered_latent_data = latent_data[mask]
print("filtered data shape:", filtered_latent_data.shape)
filtered_labels = all_labels_array[mask]
filtered_labels_neutrophil = all_labels_array[mask2]

print("filtered neutrophil label shape:", filtered_labels_neutrophil.shape)

myeloblast_indices = np.where(filtered_labels == label_map['myeloblast'])[0]
neutrophil_banded_indices = np.where(filtered_labels == label_map['neutrophil_banded'])[0]
neutrophil_seg_indices = np.where(filtered_labels == label_map['neutrophil_segmented'])[0]
basophil_indices = np.where(filtered_labels == label_map['basophil'])[0]
eosinophil_indices = np.where(filtered_labels == label_map['eosinophil'])[0]
monocyte_indices = np.where(filtered_labels == label_map['monocyte'])[0]

# np.random.seed(10)
random_myeloblast_index = np.random.choice(myeloblast_indices)
random_neutrophil_banded_index = np.random.choice(neutrophil_banded_indices)
random_neutrophil_seg_index = np.random.choice(neutrophil_seg_indices)
random_basophil_index = np.random.choice(basophil_indices)
random_eosinophil_index = np.random.choice(eosinophil_indices)
random_monocyte_index = np.random.choice(monocyte_indices)

random_myeloblast_point = filtered_latent_data[random_myeloblast_index]
# You can replace filtered_laten_data with neutrophil_data
random_neutrophil_banded_point = filtered_latent_data[random_neutrophil_banded_index]
random_neutrophil_seg_point = filtered_latent_data[random_neutrophil_seg_index]
random_basophil_point = filtered_latent_data[random_basophil_index]
random_eosinophil_point = filtered_latent_data[random_eosinophil_index]
random_monocyte_point = filtered_latent_data[random_monocyte_index]


# print("Point data shape:", random_myeloblast_point.shape)


def interpolate_gpr(latent_start, latent_end, steps=100):
    if isinstance(latent_start, torch.Tensor):
        latent_start = latent_start.detach().cpu().numpy()
    if isinstance(latent_end, torch.Tensor):
        latent_end = latent_end.detach().cpu().numpy()

    indices = np.array([0, 1]).reshape(-1, 1)

    latent_vectors = np.vstack([latent_start, latent_end])

    kernel = C(1.0, (1e-1, 1e1)) * RBF(1e-1, (1e-1, 1e1))

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(indices, latent_vectors)

    index_range = np.linspace(0, 1, steps).reshape(-1, 1)

    interpolated_latent_vectors = gpr.predict(index_range)

    return interpolated_latent_vectors


def interpolate_gif_gpr(model, filename, latent_start, latent_end, steps=100, grid_size=(10, 10), device=device):
    model_1.eval()

    interpolated_latent_points = interpolate_gpr(latent_start, latent_end, steps=steps)

    file_path = 'interpolation_1'
    torch.save(interpolated_latent_points, file_path + '_latent_points.pt')
    print(f"Interpolated latent points saved to {filename}_latent_points.pt")

    decoded_images = []
    for z in interpolated_latent_points:
        z_tensor = torch.from_numpy(z).float().to(device).unsqueeze(0)
        with torch.no_grad():
            decoded_img = model_1.decoder(z_tensor)
            decoded_img = model_1.img_decoder(decoded_img)
        decoded_images.append(decoded_img.cpu())

    while len(decoded_images) < grid_size[0] * grid_size[1]:
        decoded_images.append(torch.zeros_like(decoded_images[0]))
    decoded_images = decoded_images[:grid_size[0] * grid_size[1]]

    tensor_grid = torch.stack(decoded_images).squeeze(1)  # Remove batch dimension if necessary
    grid_image = make_grid(tensor_grid, nrow=grid_size[1], normalize=True, padding=2)
    grid_image = ToPILImage()(grid_image)
    grid_image.save(filename + '.jpg', quality=300)
    print("Grid Image saved successfully")


def get_images_from_different_classes(dataloader, class_1_label, class_2_label):
    feature_1, feature_2 = None, None

    for feature, _, _, labels, _ in dataloader:
        if feature_1 is not None and feature_2 is not None:
            break

        for i, label in enumerate(labels):
            if label.item() == class_1_label and feature_1 is None:
                feature_1 = feature[i].unsqueeze(0)

            if label.item() == class_2_label and feature_2 is None:
                feature_2 = feature[i].unsqueeze(0)

    return [feature_1, feature_2]


def get_latent_vector(x):
    distributions = model_1.encoder(x)
    print(f"Distributions shape: {distributions.shape}")
    mu = distributions[:, :50]
    logvar = distributions[:, 50:100]
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    z = reparametrize(mu, logvar)
    print("Shape of z:", z.shape)
    return z


train_dataset = Dataloader(split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)

selected_features = get_images_from_different_classes(train_dataloader, label_map['neutrophil_banded'],
                                                      label_map['neutrophil_segmented'])

start_latent, end_latent = [get_latent_vector(feature.float().to(device)) for feature in selected_features]
# interpolate_gif_gpr("interpolation_img_ge", start_latent, end_latent, steps=100, grid_size=(10, 10), device=device)
interpolate_gif_gpr(model_1, "vae_interpolation_gpr_myelo_nsegment_1", random_myeloblast_point,
                    random_neutrophil_seg_point, steps=100, grid_size=(10, 10))

# SEQUENCE DECODING and GENE EXPRESSED DETECTION
interpolated_points = torch.load('interpolation_1_latent_points.pt')

model_2.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_2.to(device)

gene_expression_profiles = []
with torch.no_grad():
    for latent_vector in interpolated_points:
        latent_vector_tensor = torch.from_numpy(latent_vector).float().to(device).unsqueeze(0)
        gene_expression = model_2.decoder(latent_vector_tensor)
        gene_expression_profiles.append(gene_expression.squeeze(0))

gen_expression = torch.stack(gene_expression_profiles).cpu().numpy()
print("gene expression shape:", gen_expression.shape)
print("visualization of trajectory for each gene started")

initial_expression = gen_expression[0, :]
final_expression = gen_expression[-1, :]
abs_diff_per_gene = np.abs(final_expression - initial_expression)

ptp_values = np.ptp(gen_expression, axis=0)
threshold = np.max(ptp_values) * 0.0

variable_genes_indices = np.where(abs_diff_per_gene > threshold)[0]
filtered_gen_expression = gen_expression[:, variable_genes_indices]
print(type(filtered_gen_expression))
print(filtered_gen_expression.shape)

plt.figure(figsize=(12, 8))

for i, gene_idx in enumerate(variable_genes_indices):
    plt.plot(filtered_gen_expression[:, i], label=f'Gene {gene_idx + 1}')

plt.xlabel('Trajectory Points')
plt.ylabel('Gene Expression')
plt.title('Gene Expression Over Trajectory')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout(rect=(0.0, 0.1, 0.75, 0.9))
plt.savefig(os.path.join(umap_dir, 'gene_expression_trajectory.png'))
plt.close()
print("trajectory is saved")

print("Calculate fold change")
small_const = 1e-10
mean_expression = np.mean(filtered_gen_expression, axis=0)

fold_changes = filtered_gen_expression / (mean_expression + small_const)

# fold_changes = np.log2(fold_changes)

plt.figure(figsize=(20, 10))

for i in range(fold_changes.shape[1]):
    plt.plot(fold_changes[:, i], label=f'Gene {i + 1}')

plt.xlabel('Trajectory Points')
plt.ylabel('Fold Change')
plt.title('Fold Change of Gene Expression Over Trajectory')
plt.xlim(left=0, right=fold_changes.shape[0] - 1)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout(rect=(0.0, 0.1, 0.75, 0.9))
plt.savefig(os.path.join(umap_dir, 'gene_expression_fold_change_trajectory.png'))
plt.close()
print("fold change is saved")

# clustering
X_train = TimeSeriesScalerMeanVariance().fit_transform(fold_changes.T)
sz = X_train.shape[1]

# Perform kShape clustering
ks = KShape(n_clusters=3, verbose=True)
y_pred = ks.fit_predict(X_train)
n_genes_in_clusters = {i: sum(y_pred == i) for i in range(5)}

for cluster, count in n_genes_in_clusters.items():
    print(f"Number of genes in cluster {cluster}: {count}")
fig_width, fig_height = 20, 15  # Width, Height in inches
plt.figure(figsize=(fig_width, fig_height))

for yi in range(3):
    plt.subplot(3, 1, 1 + yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-6, 6)
    plt.title("Cluster %d" % (yi + 1))
    plt.tight_layout()
    plt.savefig(os.path.join(umap_dir, f'gene_expression_clusters_{yi + 1}.png'))

plt.tight_layout()
plt.close()

"""
plt.title("Gene Expression Profiles by Cluster")
plt.xlabel("Time")
plt.ylabel("Fold Change")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(umap_dir, 'gene_expression_clusters.png'))
plt.close()
"""

gene_variances = np.var(gene_expression, axis=0)
top_genes_indices = np.argsort(gene_variances)[-100:]
gene_expression_df = pd.DataFrame(gene_expression[:, top_genes_indices])
gene_expression_df.to_csv('top_100_genes_expression.csv', index=False)

# Plot a heatmap for the top 100 genes
sns.heatmap(gene_expression_df.T, cmap='viridis', yticklabels=False)
plt.title('Gene Expression Changes Along Trajectory for Top 100 Genes')
plt.xlabel('Points on Trajectory')
plt.ylabel('Top 100 Genes')
plt.savefig(os.path.join(umap_dir, 'GE_Top_100_HeatMap.png'))
plt.close()

sorted_indices = np.argsort(gene_variances)[::-1]
sorted_gene_expression = gene_expression[:, sorted_indices]
gene_distances = pdist(sorted_gene_expression.T, 'euclidean')
row_linkage = linkage(gene_distances, method='average')

norm_variances = gene_variances[sorted_indices] / gene_variances[sorted_indices].max()
variance_colors = plt.cm.viridis(norm_variances)

"""  
plt.figure(figsize=(10, 10))
sns.heatmap(
    gene_expression.T,
    cmap='viridis',
    yticklabels=True
)
plt.title('Gene Expression Changes Along Trajectory')
plt.xlabel('Points on Trajectory')
plt.ylabel('Genes')
plt.savefig(os.path.join(umap_dir, 'Gene_Expression_HeatMap.png'))
plt.close()
"""
# Plotting the clustermap
sns.clustermap(sorted_gene_expression.T,
               row_linkage=row_linkage,
               col_cluster=False,
               standard_scale=1,
               row_colors=variance_colors,
               cmap='viridis',
               figsize=(10, 10))

plt.savefig(os.path.join(umap_dir, 'GE_Cluster_MAP_with_Variance.png'))
plt.close()

# Visualization Trajectory


epoch_of_gen = 290
latent_dir = 'latent_variables_GE_3'
z_dir = 'z_variables_GE_3'

class_labels_gen = [0, 1, 2]
# monocyte : class 1
# neutrophil : class 2
# basophil : class 0

class_label = 2

mean_filename = os.path.join(latent_dir, f'class_{class_label}_mean_epoch_{epoch_of_gen}.npy')
z_reference_filename = os.path.join(z_dir, f'class_{class_label}_z_epoch_{epoch_of_gen}.npy')

ref_mean_class_2 = np.load(mean_filename)
ref_z_class_2 = np.load(z_reference_filename)
interpolated_points_np = interpolated_points.cpu().numpy()

combined_data = np.vstack([neutrophil_data, ref_z_class_2])
umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
umap_embedding = umap_reducer.fit_transform(combined_data)
umap_path = umap_reducer.transform(interpolated_points_np)

split_point = neutrophil_data.shape[0]
umap_z_neutrophil = umap_embedding[:split_point, :]
umap_ref_z_class_2 = umap_embedding[split_point:, :]

plt.figure(figsize=(12, 6))
plt.scatter(umap_z_neutrophil[:, 0], umap_z_neutrophil[:, 1], s=10, label='Model Neutrophil')
plt.scatter(umap_ref_z_class_2[:, 0], umap_ref_z_class_2[:, 1], s=10, label='Reference Neutrophil', alpha=0.6)
plt.plot(umap_path[:, 0], umap_path[:, 1], 'r-', label='Trajectory')
plt.title('UMAP Visualization of Neutrophil Latent Representations (Post-Training) and Trajectory')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend()
plt.grid(False)
plt.savefig(os.path.join(umap_dir, 'umap_neutrophil_comparison_Trajectory.png'))
plt.close()

print("completed")

# Calculate differences between consecutive interpolated points
latent_diffs = np.diff(interpolated_points, axis=0)
# Compute the norm of each difference vector (you might need to adapt this depending on your data structure)
latent_diff_norms = np.linalg.norm(latent_diffs, axis=1)

# Define a threshold for what you consider a 'significant' change
change_threshold = 0.01 # Adjust this threshold as needed

# Create a mask for points that represent significant change
significant_change_mask = latent_diff_norms > change_threshold

# Since np.diff reduces the length by 1, we need to add a True at the start to keep the first point
significant_change_mask = np.insert(significant_change_mask, 0, True)

# Filter the interpolated_points and gene_expression_profiles using the mask
filtered_interpolated_points = interpolated_points[significant_change_mask]
filtered_gene_expression_profiles = [gene_expression_profiles[i] for i in range(len(gene_expression_profiles)) if significant_change_mask[i]]

# Stack the filtered gene expression profiles
filtered_gen_expression = torch.stack(filtered_gene_expression_profiles).cpu().numpy()

# Recalculate fold changes based on the filtered data
mean_expression = np.mean(filtered_gen_expression, axis=0)
fold_changes = filtered_gen_expression / (mean_expression + small_const)

# Plot the new fold change graph without the stagnant regions
plt.figure(figsize=(20, 10))
for i, gene_idx in enumerate(variable_genes_indices):
    gene_name = gene_names[gene_idx]
    plt.plot(fold_changes[:, i], label=gene_name)

# ... rest of your plotting code ...
