from sklearn.linear_model import Lars
from sklearn.metrics import accuracy_score
import pandas as pd

# Load training dataset
train_data = pd.read_csv("training.csv")

# Separate the target variable from the rest of the features
X_train = train_data.drop("ONCOGENIC", axis=1)
y_train = train_data["ONCOGENIC"]

# Initialize the LARS model with default parameters
lars_model = Lars()

# Train the model
lars_model.fit(X_train, y_train)

# Predict the target variable on the training set
y_pred = lars_model.predict(X_train)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_train, y_pred.round())

# Print the name of the model, the training loss and the accuracy
print("Model name: LARS Regression")
print("Training loss: ", lars_model.score(X_train, y_train))
print("Accuracy: ", accuracy)
