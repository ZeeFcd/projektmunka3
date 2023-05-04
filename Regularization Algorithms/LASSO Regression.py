# Import necessary libraries
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("training.csv")

# Define the input features and target variable
X = data.drop("ONCOGENIC", axis=1)
y = data["ONCOGENIC"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Lasso regression model to the training data
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = lasso.predict(X_test)

# Calculate the accuracy of the model
acc = accuracy_score(y_test, y_pred.round())

# Print the name of the model, training loss, and accuracy
print("Model name: Lasso Regression")
print("Training loss: ", lasso.score(X_train, y_train))
print("Accuracy: ", acc)
