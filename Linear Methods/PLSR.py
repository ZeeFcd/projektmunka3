# Import required libraries
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('training.csv')

# Split the data into independent and dependent variables
X = data.drop('ONCOGENIC', axis=1)
y = data['ONCOGENIC']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create PLSR model with 3 components
model = PLSRegression(n_components=3)

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the dependent variable for both training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate the R^2 scores to measure model accuracy
train_accuracy = r2_score(y_train, y_train_pred)
test_accuracy = r2_score(y_test, y_test_pred)

# Print model name, training loss, and accuracies
print("Model Name: Partial Least Squares Regression")
print("Training Loss: N/A")
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
