import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('training.csv')

# Split the data into features (X) and target (y)
X = data.drop('ONCOGENIC', axis=1)
y = data['ONCOGENIC']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a Gaussian mixture model to the training data
n_components = 2 # We only have two classes now
gmm = GaussianMixture(n_components=n_components)
gmm.fit(X_train)

# Use the GMM to transform the training and testing data
X_train_gmm = gmm.predict_proba(X_train)
X_test_gmm = gmm.predict_proba(X_test)

# Fit a linear discriminant analysis model to the transformed data
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_gmm, y_train)

# Calculate the training accuracy
train_acc = lda.score(X_train_gmm, y_train)

# Calculate the testing accuracy
test_acc = lda.score(X_test_gmm, y_test)

# Print the model name, training loss, and accuracy
print("Model Name: Mixture Discriminant Analysis")
print("Training Loss: ", lda.get_params()['priors'])
print("Training Accuracy: ", train_acc)
print("Testing Accuracy: ", test_acc)
