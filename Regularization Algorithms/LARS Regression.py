import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("training.csv")

# Separate the target variable from the features
X = data.drop("ONCOGENIC", axis=1)
y = data["ONCOGENIC"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Elastic Net Regression model
model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Predict the target variable for the training set
y_train_pred = model.predict(X_train)

# Predict the target variable for the testing set
y_test_pred = model.predict(X_test)

# Calculate the training accuracy of the model
train_accuracy = accuracy_score(y_train, y_train_pred.round())

# Calculate the testing accuracy of the model
test_accuracy = accuracy_score(y_test, y_test_pred.round())

# Print the name of the model
print("Elastic Net Regression Model")

# Print the training loss
print("Training Loss:", model.score(X_train, y_train))

# Print the training accuracy
print("Training Accuracy:", train_accuracy)

# Print the testing accuracy
print("Testing Accuracy:", test_accuracy)
