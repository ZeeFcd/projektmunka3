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
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Train the PCR model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the target variable using the trained model
y_pred = regressor.predict(X_test)

# Calculate the R-squared score to evaluate the model's accuracy
accuracy = r2_score(y_test, y_pred)

# Print the model's name, training loss (MSE), and accuracy
print("PCR model")
print("Training loss: ", regressor.score(X_train, y_train))
print("Accuracy: ", accuracy)
