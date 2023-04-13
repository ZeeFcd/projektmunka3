import numpy as np
from sklearn.linear_model import LogisticRegression

# Generate random data
np.random.seed(42) # for reproducibility
X = np.random.randn(100, 3) # 100 samples with 3 features
y = np.random.randint(0, 2, size=100) # binary target variable

# Split data into training and testing sets
split = int(len(X) * 0.8) # 80% train, 20% test
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Create and train logistic regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# Predict on test set
y_pred = lr_model.predict(X_test)

# Evaluate model performance
accuracy = lr_model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
