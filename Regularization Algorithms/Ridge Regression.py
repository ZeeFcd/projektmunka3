import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training dataset
data = pd.read_csv('training.csv')

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('ONCOGENIC', axis=1), data['ONCOGENIC'], test_size=0.3, random_state=42)

# Create a Ridge Regression model
model = Ridge(alpha=1.0)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance on the test data
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)




