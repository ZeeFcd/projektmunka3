import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# load data
data = pd.read_csv('training.csv')

# Split the data into training and testing sets:
X = data.drop('ONCOGENIC', axis=1)
y = data['ONCOGENIC']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the Elastic Net model
model = ElasticNet(alpha=0.5, l1_ratio=0.5)

# Fit the model to the training data
model.fit(X_train, y_train)

# Predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)





