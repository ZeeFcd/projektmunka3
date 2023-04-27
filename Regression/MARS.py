# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from pyearth import Earth

# Load the CSV dataset using pandas
dataset = pd.read_csv('training.csv')

# Split the dataset into training and testing data
train_data = dataset.sample(frac=0.8, random_state=0)
test_data = dataset.drop(train_data.index)


# Generate random data for training and testing
X_train = train_data.drop(['ONCOGENIC'], axis=1).to_numpy()
y_train = train_data['ONCOGENIC'].to_numpy()
X_test = test_data.drop(['ONCOGENIC'], axis=1).to_numpy()
y_test = test_data['ONCOGENIC'].to_numpy()

# Fitting the MARS regression model
model = Earth(max_degree=2, penalty=3.0, endspan=5, verbose=True)
model.fit(X_train, y_train)

# Predicting on the testing set
y_pred = model.predict(X_test)

# Calculating the model's accuracy
score = model.score(X_test, y_test)

print("MARS Regression Model score: {:.3f}".format(score))

