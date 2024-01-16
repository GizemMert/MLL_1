import gzip
import pickle
import os

# Define the path to your .dat.gz file
file_path = "/lustre/groups/aih/raheleh.salehi/Master-thesis/Aug_features_datasets/Augmented-MLL-AML_MLLdataset.dat.gz"

# Open the gzip compressed file
with gzip.open(os.path.join(file_path), "rb") as f:
    data = pickle.load(f)

# Check the type of the loaded data
print(type(data))

# If it is a dictionary, iterate over its keys
if isinstance(data, dict):
    print("Keys in the file:", data.keys())

    # For each key, check the type of its value and print some of its contents
    for key in data.keys():
        print(f"Type of data under key '{key}': {type(data[key])}")

        # Check if the value is also a dictionary
        if isinstance(data[key], dict):
            print(f"Contents of the nested dictionary for key '{key}':")
            for nested_key, nested_value in data[key].items():
                print(f"  {nested_key}: {nested_value}")
                # Break after a few items to avoid too much output
                break

