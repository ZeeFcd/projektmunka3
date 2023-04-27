import numpy as np
import pandas as pd
from numpy.linalg import svd

# Load data
df = pd.read_csv('training.csv')

# Center data
X = df.values
X = X - np.mean(X, axis=0)

# Perform SVD
U, s, V = svd(X, full_matrices=False)

# Reconstruct data
X_reconstructed = U @ np.diag(s) @ V

# Print original and reconstructed data
print("Original data:\n", X)
print("\nReconstructed data:\n", X_reconstructed)


