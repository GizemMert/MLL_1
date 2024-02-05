import networkx as nx
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from PIL import Image
import os
from matplotlib.gridspec import GridSpec
from model4 import VariationalAutoencodermodel4, reparametrize
from Dataloader_2 import Dataloader
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import geomstats.backend as gs
import networkx as nx
import torch
import numpy as np
import cv2
import umap
import matplotlib.pyplot as plt
from geomstats.information_geometry.normal import NormalDistributions
import geomstats.geometry.complex_manifold as cm

# dimension = 30
# complex_manifold = cm.ComplexManifold(dimension)

normal = NormalDistributions(sample_dim=1)
epoch = 140
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
inverse_label_map = {v: k for k, v in label_map.items()}  # inverse mapping for UMAP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoencodermodel4(latent_dim=30)
model_save_path = 'trained_model4cp2_new5.pth'
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()

umap_dir = 'umap_path'
if not os.path.exists(umap_dir):
    os.makedirs(umap_dir)

# Load all latent representations
latent_dir = 'latent_data4cp2_new5'
latents_path = os.path.join(latent_dir, f'latent_epoch_{epoch}.npy')
label_dir = 'label_data4cp2_new5'
labels_path = os.path.join(label_dir, f'label_epoch_{epoch}.npy')

# Load all latent representations
latent_data = np.load(latents_path)
latent_data_reshaped = latent_data.reshape(latent_data.shape[0], -1)
print("Latent data shape:", latent_data_reshaped.shape)

# Load all labels
all_labels_array = np.load(labels_path)
print("Labels array shape:", all_labels_array.shape)

# print("Labels array shape:", all_labels_array.shape)

# Filter out the 'erythroblast' class
erythroblast_class_index = label_map['erythroblast']
mask = all_labels_array != erythroblast_class_index
filtered_latent_data = latent_data[mask]
print("filtered data shape:", filtered_latent_data.shape)
filtered_labels = all_labels_array[mask]

myeloblast_indices = np.where(filtered_labels == label_map['myeloblast'])[0]
neutrophil_banded_indices = np.where(filtered_labels == label_map['neutrophil_banded'])[0]

# np.random.seed(42)
random_myeloblast_index = np.random.choice(myeloblast_indices)
random_neutrophil_banded_index = np.random.choice(neutrophil_banded_indices)

random_myeloblast_point = filtered_latent_data[random_myeloblast_index]
random_neutrophil_banded_point = filtered_latent_data[random_neutrophil_banded_index]
print("Poin data shape:", random_myeloblast_point.shape)

def compute_centers_of_mass(latent_data, labels):
    unique_labels = np.unique(labels)
    centers_of_mass = {}
    for label in unique_labels:
        class_points = latent_data[labels == label]
        center_of_mass = np.mean(class_points, axis=0)
        centers_of_mass[label] = center_of_mass
    return centers_of_mass


def construct_graph(centers_of_mass, start_latent, end_latent):
    G = nx.Graph()
    G.add_node('start', pos=start_latent)
    G.add_node('end', pos=end_latent)

    # Add centers of mass as nodes
    for label, center in centers_of_mass.items():
        G.add_node(label, pos=center)

    # Compute and add edges based on distances
    for node1 in G.nodes:
        for node2 in G.nodes:
            if node1 != node2:
                dist = np.linalg.norm(G.nodes[node1]['pos'] - G.nodes[node2]['pos'])
                G.add_edge(node1, node2, weight=dist)

    return G

def find_shortest_path(graph):
    path = nx.dijkstra_path(graph, source='start', target='end', weight='weight')
    return path

def interpolate_gif_dijkstra(filename, start_latent, end_latent, latent_dataset, labels):
    model.eval()

    # Compute centers of mass for each class
    centers_of_mass = compute_centers_of_mass(latent_dataset, labels)

    # Adding start and end latents to the centers of mass
    centers_of_mass['start'] = start_latent.detach().cpu().numpy()
    centers_of_mass['end'] = end_latent.detach().cpu().numpy()

    # Construct the graph
    G = construct_graph(centers_of_mass, start_latent, end_latent)

    # Find the shortest path using Dijkstra's algorithm
    shortest_path_labels = find_shortest_path(G, 'start', 'end')

    # For each label in the shortest path, get the corresponding latent vector
    path_latents = np.array([centers_of_mass[label] for label in shortest_path_labels if label in centers_of_mass])

    num_points = len(path_latents)

    # Desired number of rows, adjust as needed
    num_rows = 10  # For example

    # Calculate the number of columns needed
    num_columns = int(np.ceil(num_points / num_rows))

    # Update grid_size based on the number of points
    grid_size = (num_rows, num_columns)

    # Decode each point along this path to generate images
    decoded_images = []
    for z in path_latents:
        z_tensor = torch.from_numpy(z).float().to(device).unsqueeze(0)
        with torch.no_grad():
            decoded_img = model.decoder(z_tensor)
            decoded_img = model.img_decoder(decoded_img)
        decoded_images.append(decoded_img.cpu())

    # Ensure the decoded images fit into the specified grid size
    while len(decoded_images) < grid_size[0] * grid_size[1]:
        decoded_images.append(torch.zeros_like(decoded_images[0]))
    decoded_images = decoded_images[:grid_size[0] * grid_size[1]]

    # Arrange images in a grid and save
    tensor_grid = torch.stack(decoded_images).squeeze(1)  # Remove batch dimension if necessary
    grid_image = make_grid(tensor_grid, nrow=grid_size[1], normalize=True, padding=2)
    grid_image = ToPILImage()(grid_image)
    grid_image.save(filename + '.jpg', quality=95)
    print("Image saved successfully")


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
    distributions = model.encoder(x)
    mu = distributions[:, :latent_dim]
    logvar = distributions[:, latent_dim:]
    z = reparametrize(mu, logvar)
    return z


train_dataset = Dataloader(split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)

selected_features = get_images_from_different_classes(train_dataloader, label_map['myeloblast'], label_map['neutrophil_banded'])

start_latent, end_latent = [get_latent_vector(feature.float().to(device)) for feature in selected_features]

interpolate_gif_dijkstra("vae_interpolation_COM", start_latent=start_latent, end_latent=end_latent, latent_dataset=filtered_latent_data, labels=filtered_labels)


"""
    total_slots = grid_size[0] * grid_size[1]
    while len(decoded_images) < total_slots:
        decoded_images.append(torch.zeros_like(decoded_images[0]))

    # Trim the list to match the grid size exactly
    decoded_images = decoded_images[:total_slots]

    # Arrange images in a grid
    tensor_grid = torch.stack(decoded_images).squeeze(1)  # Remove batch dimension if necessary
    grid_image = make_grid(tensor_grid, nrow=grid_size[1], normalize=True, padding=2)
    grid_image = ToPILImage()(grid_image)
    grid_image.save(filename + '.jpg', quality=95)
    print("Image saved successfully")


def get_latent_vector(x):
    distributions = model.encoder(x)
    mu = distributions[:, :latent_dim]
    logvar = distributions[:, latent_dim:]
    z = reparametrize(mu, logvar)
    return z

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


train_dataset = Dataloader(split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)

selected_features = get_images_from_different_classes(train_dataloader, label_map['myeloblast'], label_map['neutrophil_banded'])

start_latent, end_latent = [get_latent_vector(feature.float().to(device),) for feature in selected_features]

interpolate_gif_gpr("vae_interpolation_gpr", random_myeloblast_point, random_neutrophil_banded_point, steps=100, grid_size=(10, 10))


interpolated_latents = interpolate_gpr(random_myeloblast_point, random_neutrophil_banded_point, n_points=100)

combined_data = np.vstack([filtered_latent_data, interpolated_latents])
# UMAP for latent space
umap_model = umap.UMAP(n_neighbors=13, min_dist=0.1, n_components=2, metric='euclidean')
combined_data_umap = umap_model.fit_transform(combined_data)
interpolated_latents_umap = combined_data_umap[-100:]
latent_data_umap = combined_data_umap[:-100]



fig = plt.figure(figsize=(12, 10), dpi=150)
gs = GridSpec(1, 2, width_ratios=[4, 1], figure=fig)

ax = fig.add_subplot(gs[0])
scatter = ax.scatter(latent_data_umap[:, 0], latent_data_umap[:, 1], s=100, c=filtered_labels, cmap='Spectral')

ax.plot(interpolated_latents_umap[:, 0], interpolated_latents_umap[:, 1], color='black', linestyle='-', linewidth=5)
ax.set_aspect('equal')

x_min, x_max = np.min(latent_data_umap[:, 0]), np.max(latent_data_umap[:, 0])
y_min, y_max = np.min(latent_data_umap[:, 1]), np.max(latent_data_umap[:, 1])

zoom_factor = 0.40  # Smaller values mean more zoom
padding_factor = 0.3  # Adjust padding around the zoomed area

# Calculate the range for zooming in based on the zoom factor
x_range = (x_max - x_min) * zoom_factor
y_range = (y_max - y_min) * zoom_factor

# Calculate the center of the data
center_x = (x_max + x_min) / 2
center_y = (y_max + y_min) / 2

# Calculate new limits around the center of the data
new_x_min = center_x - (x_range * (1 + padding_factor))
new_x_max = center_x + (x_range * (1 + padding_factor))
new_y_min = center_y - (y_range * (1 + padding_factor))
new_y_max = center_y + (y_range * (1 + padding_factor))

# Apply the new limits to zoom in on the plot
ax.set_xlim(new_x_min, new_x_max)
ax.set_ylim(new_y_min, new_y_max)

ax.set_title(f'Latent Space Representation - (Epoch {epoch})', fontsize=18)
ax.set_xlabel('UMAP Dimension 1', fontsize=16)
ax.set_ylabel('UMAP Dimension 2', fontsize=16)

# Second subplot for the legend
ax_legend = fig.add_subplot(gs[1])
ax_legend.axis('off')  # Turn off the axis for the legend subplot

unique_filtered_labels = np.unique(filtered_labels)
filtered_class_names = [inverse_label_map[label] for label in unique_filtered_labels if label in inverse_label_map]
color_map = plt.cm.Spectral(np.linspace(0, 1, len(unique_filtered_labels)))

legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=filtered_class_names[i],
                             markerfacecolor=color_map[i], markersize=18) for i in range(len(filtered_class_names))]

ax_legend.legend(handles=legend_handles, loc='center', fontsize=16, title='Cell Types')


plt.tight_layout()
umap_figure_filename = os.path.join(umap_dir, f'umap_epoch_{epoch}.png')
plt.savefig(umap_figure_filename, bbox_inches='tight', dpi=300)
print("umap_path is saved")
plt.close(fig)


plt.hist(latent_data_reshaped.flatten(), bins=30, density=True, alpha=0.6, color='g')
plt.title("Histogram of Latent Data")
plt.savefig("latent_data_histogram.png")  # Save histogram
plt.close()  # Close the plot

# Q-Q plot
stats.probplot(filtered_latent_data.flatten(), dist="norm", plot=plt)
plt.title("Q-Q Plot of Latent Data")
plt.savefig("latent_data_qqplot.png")  # Save Q-Q plot
plt.close()  # Close the plot

# Shapiro-Wilk Test
shapiro_test = stats.shapiro(filtered_latent_data.flatten())
print("Shapiro-Wilk Test: ", shapiro_test)

# Kolmogorov-Smirnov Test
ks_test = stats.kstest(filtered_latent_data.flatten(), 'norm',
                       args=(filtered_latent_data.mean(),filtered_latent_data.std()))
print("Kolmogorov-Smirnov Test: ", ks_test)

mean = np.mean(filtered_latent_data, axis=0)
covariance_matrix = np.cov(filtered_latent_data, rowvar=False)

# Check for centered distribution
is_centered = np.allclose(mean, 0)

# Check for diagonal distribution
is_diagonal = np.allclose(covariance_matrix, np.diag(np.diagonal(covariance_matrix)))

# Determine distribution type
if is_centered:
    distribution_type = 'centered'
elif is_diagonal:
    distribution_type = 'diagonal'
else:
    distribution_type = 'general'

print("Distribution type:", distribution_type)



distributions = {
    'expon': stats.expon,
    'gamma': stats.gamma,
}

for name, dist in distributions.items():
    if name in ['expon', 'gamma'] and not np.issubdtype(filtered_latent_data.dtype, float):
        continue

    params = dist.fit(filtered_latent_data)

    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Calculate fitted PDF and error with fit in distribution
    sorted_data = np.sort(filtered_latent_data)
    if arg:
        pdf = dist.pdf(sorted_data, *arg, loc=loc, scale=scale)
    else:
        pdf = dist.pdf(sorted_data, loc=loc, scale=scale)

    # Calculate the log likelihood for the fitted distribution
    log_likelihood = np.sum(dist.logpdf(filtered_latent_data, *arg, loc=loc, scale=scale))

    # Plot the histogram and PDF
    plt.figure(figsize=(12, 8))
    plt.hist(filtered_latent_data, bins=30, density=True, alpha=0.6, color='g', label='Data histogram')
    plt.plot(sorted_data, pdf, label=f'{name} fit (LL={log_likelihood:.2f})')
    plt.title(f'Fit of {name} distribution')
    plt.xlabel('Data')
    plt.ylabel('Frequency')
    plt.legend()
    plot_filename = f"{name}_distribution_fit.png"  # Unique filename for each plot
    plt.savefig(plot_filename)
    plt.close()

    print(f"Plot saved as {plot_filename}")

n = np.max(filtered_latent_data)  # This is just an example, adjust it as needed
p_est = np.mean(filtered_latent_data) / n
binom_est = stats.binom(n=n, p=p_est)

print(f"Estimated parameters for Binomial distribution: n = {n}, p = {p_est}")

lambda_est = np.mean(filtered_latent_data)
poisson_est = stats.poisson(mu=lambda_est)

print(f"Estimated parameter for Poisson distribution: λ = {lambda_est}")

"""