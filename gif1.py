from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from PIL import Image
import torch
import numpy as np
from model4 import VariationalAutoencodermodel4, reparametrize
from Dataloader_2 import Dataloader
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import geomstats.backend as gs
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoencodermodel4(latent_dim=30)
model_save_path = '/Users/gizem/MLL_1/trained_model4cp2_new5 (1).pth'
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()


# Load all latent representations
latent_dir = 'latent_data4cp2_new5'
latents_path = '/Users/gizem/MLL_1/latent_epoch_140 (1).npy'
label_dir = 'label_data4cp2_new5'
labels_path = '/Users/gizem/MLL_1/label_epoch_140 (1).npy'

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

def interpolate_gpr(latent_start, latent_end, n_points=20):
    if isinstance(latent_start, torch.Tensor):
        latent_start = latent_start.detach().cpu().numpy()
    if isinstance(latent_end, torch.Tensor):
        latent_end = latent_end.detach().cpu().numpy()

    indices = np.array([0, 1]).reshape(-1, 1)


    latent_vectors = np.vstack([latent_start, latent_end])

    kernel = C(1.0, (1e-1, 1e1)) * RBF(1e-1, (1e-1, 1e1))

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(indices, latent_vectors)

    index_range = np.linspace(0, 1, n_points).reshape(-1, 1)

    interpolated_latent_vectors = gpr.predict(index_range)

    return interpolated_latent_vectors

def interpolate_gif_gpr(filename, start_latent, end_latent, steps=100, grid_size=(10, 10)):
    model.eval()

    # Compute interpolated latent vectors using GPR
    interpolated_latents = interpolate_gpr(start_latent, end_latent, steps)

    decoded_images = []
    for i, z in enumerate(interpolated_latents):
        z_tensor = torch.from_numpy(z).float().to(device).unsqueeze(0)
        with torch.no_grad():
            decoded_img = model.decoder(z_tensor)
            decoded_img = model.img_decoder(decoded_img)
        decoded_images.append(decoded_img.cpu())

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
"""
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