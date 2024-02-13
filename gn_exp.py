import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata
import importlib
from sklearn.decomposition import PCA

import matplotlib as mpl

from sklearn.neighbors import NearestNeighbors
mpl.rcParams['figure.dpi'] = 300
import sys

data = 'TS_Blood.h5ad'
adata_orig = sc.read_h5ad(data)
print(adata_orig)
