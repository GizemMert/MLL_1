import os

directory_path = '/lustre/groups/aih/raheleh.salehi/Master-thesis/MRCNN-leukocyte/data/mask_AML_MLL'

num_files = len([name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])

print(f"The number of images in the directory is: {num_files}")

