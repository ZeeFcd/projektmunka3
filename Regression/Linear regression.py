import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("training.csv")

# Split the dataset into X (features) and y (target)
X = data.drop("ONCOGENIC", axis=1)
y = data["ONCOGENIC"]

# Create a linear regression object
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions on the training data
y_pred = model.predict(X)

# Calculate training loss (mean squared error)
training_loss = mean_squared_error(y, y_pred)

# Calculate the accuracy of the model (R-squared)
accuracy = r2_score(y, y_pred)

# Print the name of the model
print("Linear Regression")

# Print the training loss
print("Training loss:", training_loss)

# Print the accuracy of the model
print("Accuracy:", accuracy)
