import numpy as np
import os

epoch = 290

# Directories where the files are saved
latent_dir = 'latent_variables_GE_3'
z_dir = 'z_variables_GE_3'
log_dir = 'log_GE_3'


class_labels = [0, 1, 2]

for class_label in class_labels:

    mean_filename = os.path.join(latent_dir, f'class_{class_label}_mean_epoch_{epoch}.npy')
    z_filename = os.path.join(z_dir, f'class_{class_label}_z_epoch_{epoch}.npy')
    log_filename = os.path.join(log_dir, f'class_{class_label}_log_epoch_{epoch}.npy')


    if os.path.exists(mean_filename):
        class_means = np.load(mean_filename)
        print(f"Epoch {epoch}, Class {class_label}, Mean file shape: {class_means.shape}")
    else:
        print(f"Mean file for epoch {epoch}, class {class_label} does not exist.")

    if os.path.exists(z_filename):
        class_z = np.load(z_filename)
        print(f"Epoch {epoch}, Class {class_label}, Z file shape: {class_z.shape}")
    else:
        print(f"Z file for epoch {epoch}, class {class_label} does not exist.")

    if os.path.exists(log_filename):
        class_log = np.load(log_filename)
        print(f"Epoch {epoch}, Class {class_label}, Log file shape: {class_log.shape}")
    else:
        print(f"Log file for epoch {epoch}, class {class_label} does not exist.")
