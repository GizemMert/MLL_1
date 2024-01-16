import gzip
import pickle

# Path to the gzipped pickle file
file_path = '/lustre/groups/aih/raheleh.salehi/Master-thesis/save_files/mll_images.pkl.gz'

# Open the gzip file and load its contents
with gzip.open(file_path, 'rb') as f:
    data = pickle.load(f)

# Now, count the number of images based on the structure of `data`
if isinstance(data, dict):
    # If the data is a dictionary, count the number of keys
    num_images = len(data.keys())
    print(f"The number of images in the file is: {num_images}")
else:
    # If the data is not a dictionary, adapt this part based on its structure
    print("Data is not in expected dictionary format, please check its structure.")
