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

# Make predictions on the training and testing data
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

# Calculate the training and testing accuracy of the model
train_accuracy = accuracy_score(y_train, y_train_pred.round())
test_accuracy = accuracy_score(y_test, y_test_pred.round())

# Print the name of the model
print("Model name: Lasso Regression")

# Print the training loss
print("Training loss:", lasso.score(X_train, y_train))

# Print the training accuracy
print("Training Accuracy:", train_accuracy)

# Print the testing accuracy
print("Testing Accuracy:", test_accuracy)
