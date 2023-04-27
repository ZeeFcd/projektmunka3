import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis

data = pd.read_csv("training.csv")

X = data.drop(columns=["ONCOGENIC"])

# Create an instance of the FactorAnalysis class and specify the number of latent factors we want to extract
fa = FactorAnalysis(n_components=3)

# Fit the factor analysis model to our data
fa.fit(X)

# Extract the factor loadings
loadings = pd.DataFrame(fa.components_, columns=X.columns)

# Extract the factor scores for each observation in our dataset
scores = pd.DataFrame(fa.transform(X))

# Extracting the eigenvalues
eigenvalues = fa.get_eigenvalues()

# Extracting the communality estimates
communality = fa.get_communalities()

# Extracting the variance explained by each factor
variance = fa.get_factor_variance()

# Printing the results
print("Eigenvalues:\n", eigenvalues)
print("Communality estimates:\n", communality)
print("Variance explained:\n", variance)