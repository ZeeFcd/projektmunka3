# Import required libraries
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv('training.csv')

# Split the data into independent and dependent variables
X = data.drop('ONCOGENIC', axis=1)
y = data['ONCOGENIC']

# Create PLSR model with 3 components
model = PLSRegression(n_components=3)

# Fit the model to the training data
model.fit(X, y)

# Predict the dependent variable using the model
y_pred = model.predict(X)

# Calculate the R^2 score to measure model accuracy
accuracy = r2_score(y, y_pred)

# Print model name, training loss and accuracy
print("Model Name: Partial Least Squares Regression")
print("Training Loss: N/A")
print("Accuracy:", accuracy)
