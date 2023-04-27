import pandas as pd
from sklearn.decomposition import NMF

# Load the training dataset into a pandas DataFrame
df = pd.read_csv("training.csv")

# Extract the data as a numpy array and target variable
X = df.drop("ONCOGENIC", axis=1).values
y = df["ONCOGENIC"].values

# Define the number of components for NMF
n_components = 5

# Instantiate an NMF object with the specified number of components
model = NMF(n_components=n_components)

# Fit the model to the data
W = model.fit_transform(X)
H = model.components_

# Print the shape of the factorized matrices
print("W shape:", W.shape)
print("H shape:", H.shape)

