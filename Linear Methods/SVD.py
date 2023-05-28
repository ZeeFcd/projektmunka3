import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('training.csv')

# Extract the features and target variable
X = df.drop('ONCOGENIC', axis=1)
y = df['ONCOGENIC']

# Perform SVD on the feature matrix
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_svd, y, test_size=0.2, random_state=42)

# Train a logistic regression model on the training set
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict on the training set and calculate the training accuracy
y_train_pred = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Predict on the testing set and calculate the testing accuracy
y_test_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the name of the model, training accuracy, and testing accuracy
print("Model Name: SVD + Logistic Regression")
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
