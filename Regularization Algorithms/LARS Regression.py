import pandas as pd
from sklearn.linear_model import Lars

# Load the training dataset
data = pd.read_csv("training.csv")

# Split the dataset into features (X) and target variable (y)
X = data.drop(columns=["ONCOGENIC"])
y = data["ONCOGENIC"]

# Create a LARS regression model
model = Lars()

# Fit the model to the training data
model.fit(X, y)

# Predict the target variable for a new data point
new_data = pd.DataFrame({"feature_1": [1], "feature_2": [2], "feature_3": [3]})
prediction = model.predict(new_data)

print(prediction)

