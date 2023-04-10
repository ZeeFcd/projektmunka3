from sklearn import *
import pandas as pd
import numpy as np
from pyearth import Earth

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

# ------------- Regression Models -----------------

# Linear Regression Model
model = LinearRegression()

# ------ Ordinary Least Squares --------
model.fit(x, y)

x_test = np.array([[3, 5], [4, 6]])
y_pred = model.predict(x_test)

print(y_pred)


# ------ Logistic Regression --------

# Splitting
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

# Creating a logistic regression model and fit it to the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing data
score = model.score(X_test, y_test)
print(f"Model accuracy: {score}")


# ------ Stepwise Regression --------

# Use RFE with stepwise selection to select the best features
selector = RFE(model, step=1)
selector = selector.fit(x, y)

# Print the selected features and their ranking
print("Selected Features:")
for i in range(len(selector.support_)):
    if selector.support_[i]:
        print(f"Feature {i+1}, rank {selector.ranking_[i]}")


# ------ Multivariate Adaptive Regression Splines (MARS) --------

# Fit a MARS model to the training data
# The PyEarth library provides an implementation of MARS that is optimized for performance and has additional features such as automatic feature selection and interaction detection
model = Earth(max_degree=2, max_terms=10, penalty=3.0, endspan=5)
model.fit(X_train, y_train)

# Evaluate the model on the testing data
score = model.score(X_test, y_test)
print(f"R^2 score: {score}")



# ------ Locally Estimated Scatterplot Smoothing (LOESS) --------
# Fit LOESS model
lowess = sm.nonparametric.lowess(y, x, frac=0.3)

# extract the smoothed x and y values
y_smoothed = lowess[:, 1]
x_smoothed = lowess[:, 0]