import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv("training.csv")

# Split the data into input features and target variable
X = data.drop('ONCOGENIC', axis=1)
y = data['ONCOGENIC']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform PCA to reduce the number of input features
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train the PCR model
regressor = LinearRegression()
regressor.fit(X_train_pca, y_train)

# Predict the target variable using the trained model
y_train_pred = regressor.predict(X_train_pca)
y_test_pred = regressor.predict(X_test_pca)

# Calculate the R-squared scores to evaluate the model's accuracy
train_accuracy = r2_score(y_train, y_train_pred)
test_accuracy = r2_score(y_test, y_test_pred)

# Print the model's name, training and testing accuracy
print("PCR model")
print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)
