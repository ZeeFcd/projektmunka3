import pandas as pd
from pyearth import Earth

# Load the dataset
data = pd.read_csv('training.csv')

# Split the dataset into features and target
X = data.drop(columns=['ONCOGENIC'])
y = data['ONCOGENIC']

# Create and train the MARS model
model = Earth(max_terms=10, max_degree=3, penalty=3.0)
model.fit(X, y)

# Calculate the accuracy of the model
accuracy = model.score(X, y)

# Print the name of the model
print("MARS model using py-earth")

# Print the training loss
print("Training loss: ", model.mse_)

# Print the accuracy of the model
print("Accuracy: ", accuracy)
