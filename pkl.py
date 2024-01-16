import gzip
import pickle
import os

# Define the path to your pickle file
images_path = "/lustre/groups/aih/raheleh.salehi/Master-thesis/save_files/mll_images.pkl.gz"

# Open the gzip compressed pickle file
with gzip.open(os.path.join(images_path), "rb") as f:
    data = pickle.load(f)

# Now `data` contains the contents of the pickle file.
# You can print out its type to understand if it's a list, dict, etc.
print(type(data))

# If it is a dictionary, you can check its keys
if isinstance(data, dict):
    print("Keys in the pickle file:", data.keys())

    # Optionally, you can print the type of the values associated with each key
    for key in data.keys():
        print(f"Type of data under key '{key}': {type(data[key])}")

    # If you want to see the first few entries under each key (for example, if they are lists or dicts)
    for key in data.keys():
        print(f"First few entries for key '{key}': {data[key][:5]}")
