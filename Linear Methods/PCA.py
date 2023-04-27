import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('training.csv')

# Separate the target variable from the features
target = data['ONCOGENIC']
features = data.drop('ONCOGENIC', axis=1)

# Standardize the features
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Apply PCA to the standardized features
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_std)

# Create a new DataFrame with the principal components and target variable
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
principal_df['ONCOGENIC'] = target

# Plot the principal components
import matplotlib.pyplot as plt
plt.scatter(principal_df['PC1'], principal_df['PC2'], c=principal_df['target'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

