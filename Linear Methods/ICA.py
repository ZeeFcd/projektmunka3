import pandas as pd
from sklearn.decomposition import FastICA

# Load dataset
data = pd.read_csv('training.csv')

# Separate features and target
X = data.drop('ONCOGENIC', axis=1)
y = data['ONCOGENIC']

# Instantiate FastICA with 10 components
ica = FastICA(n_components=10)

# Fit the model to the data
X_ica = ica.fit_transform(X)

# Print the independent components
print(X_ica)

