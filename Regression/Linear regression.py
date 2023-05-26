import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("training.csv")

# Split the dataset into X (features) and y (target)
X = data.drop("ONCOGENIC", axis=1)
y = data["ONCOGENIC"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression object
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate training loss (mean squared error)
training_loss = mean_squared_error(y_train, y_train_pred)

# Calculate testing loss (mean squared error)
testing_loss = mean_squared_error(y_test, y_test_pred)

# Calculate the training accuracy of the model (R-squared)
training_accuracy = r2_score(y_train, y_train_pred)

# Calculate the testing accuracy of the model (R-squared)
testing_accuracy = r2_score(y_test, y_test_pred)

# Print the name of the model
print("Linear Regression")

# Print the training loss and accuracy
print("Training Loss:", training_loss)
print("Training Accuracy:", training_accuracy)

# Print the testing loss and accuracy
print("Testing Loss:", testing_loss)
print("Testing Accuracy:", testing_accuracy)
