import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate random data for training and testing
X_train = np.random.rand(100, 1)
y_train = 2*X_train + np.random.randn(100, 1)*0.2
X_test = np.random.rand(20, 1)
y_test = 2*X_test + np.random.randn(20, 1)*0.2

# Create a Linear Regression object
regressor = LinearRegression()

# Train the model using the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Compute mean squared error on the test data
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)
