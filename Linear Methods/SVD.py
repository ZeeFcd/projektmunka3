import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score

# Load the training dataset
df = pd.read_csv('training.csv')

# Extract the features and target variable
X = df.drop('ONCOGENIC', axis=1)
y = df['ONCOGENIC']

# Perform SVD on the feature matrix
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)

# Train a simple logistic regression model on the reduced feature matrix
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_svd, y)

# Calculate the training accuracy
y_pred = clf.predict(X_svd)
accuracy = accuracy_score(y, y_pred)

# Print the name of the model, training loss, and accuracy
print("Model Name: SVD + Logistic Regression")
print("Training Loss: N/A")
print("Training Accuracy:", accuracy)
