from Dataloader_2 import equivalent_classes
import os
import pickle
import gzip
import random
def create_subset(features_path, images_path, num_samples):
    # Determine the directory of the dataloader.py script
    dataloader_dir = os.path.dirname(os.path.realpath(__file__))

    # Define paths for the subset directories
    subset_features_dir = os.path.join(dataloader_dir, 'subset_features1')
    subset_images_dir = os.path.join(dataloader_dir, 'subset_images1')

    # Check and create directories if they don't exist
    if not os.path.exists(subset_features_dir):
        os.makedirs(subset_features_dir)
    if not os.path.exists(subset_images_dir):
        os.makedirs(subset_images_dir)

    # Load the entire features dataset
    with gzip.open(features_path, "rb") as f:
        features = pickle.load(f)

    # Load the entire images dataset
    with gzip.open(images_path, "rb") as f:
        images = pickle.load(f)

    # Create a list of keys that have labels in equivalent_classes
    keys_with_labels = [k for k in features.keys() if features[k].get("label") in equivalent_classes]

    # Randomly choose num_samples samples
    chosen_keys = random.sample(keys_with_labels, num_samples)

    # Extracting and saving subset of features
    subset_features = {k: features[k] for k in chosen_keys}
    with gzip.open(os.path.join(subset_features_dir, 'subset_features1.dat.gz'), 'wb') as f:
        pickle.dump(subset_features, f)

    # Extracting and saving subset of images
    subset_images = {k: images[k] for k in chosen_keys}
    with gzip.open(os.path.join(subset_images_dir, 'subset_images1.pkl.gz'), 'wb') as f:
        pickle.dump(subset_images, f)

    print(f"Features subset saved in {subset_features_dir}")

    print(f"Images subset saved in {subset_images_dir}")

# Usage
features_mll_path = "/lustre/groups/aih/raheleh.salehi/Master-thesis/Aug_features_datasets/Augmented-MLL-AML_MLLdataset.dat.gz"
images_path = "/lustre/groups/aih/raheleh.salehi/Master-thesis/save_files/mll_images.pkl.gz"
create_subset(features_mll_path, images_path, 30)
