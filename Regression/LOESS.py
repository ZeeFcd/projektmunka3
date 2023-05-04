import pandas as pd
import statsmodels.api as sm

# Load dataset
data = pd.read_csv("training.csv")

# Define predictor and response variables
X = data.drop(columns=["ONCOGENIC"])
y = data["ONCOGENIC"]

# Fit LOESS model
lowess = sm.nonparametric.lowess
z = lowess(y, X.values[:, 0], frac=0.3)

# Calculate accuracy of model
y_pred = z[:, 1] >= 0.5
accuracy = sum(y == y_pred) / len(y)

# Print model information
print("Model: LOESS")
print("Training Loss:", sum((y - z[:, 1]) ** 2))
print("Accuracy:", accuracy)
