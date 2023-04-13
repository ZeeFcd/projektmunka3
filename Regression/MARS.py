# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from pyearth import Earth

# Generating random data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fitting the MARS regression model
model = Earth(max_degree=2, penalty=3.0, endspan=5, verbose=True)
model.fit(X_train, y_train)

# Predicting on the testing set
y_pred = model.predict(X_test)

# Calculating the model's accuracy
score = model.score(X_test, y_test)

print("MARS Regression Model score: {:.3f}".format(score))

