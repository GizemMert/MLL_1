import gzip

# Path to your gzipped file
file_path = '/lustre/groups/aih/raheleh.salehi/Master-thesis/Aug_features_datasets/Augmented-MLL-AML_MLLdataset.dat.gz'

# Open and read the contents of the gzipped file
with gzip.open(file_path, 'rb') as f:
    file_content = f.read()

# At this point, you need to know the format of the data inside the .dat file
# If it's a text file or a pickle file, you can proceed accordingly

# For example, if it's a pickle file, you would do:
import pickle
data = pickle.loads(file_content)
