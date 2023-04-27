import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv('training.csv')

# Split the dataset into features and target variable
X = df.drop('ONCOGENIC', axis=1)
y = df['ONCOGENIC']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the PLSR model with 2 components
pls = PLSRegression(n_components=2)

# Fit the PLSR model to the training data
pls.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = pls.predict(X_test)

# Calculate the R-squared score for the predictions
r2 = r2_score(y_test, y_pred)

# Print the R-squared score
print('R-squared score: {:.2f}'.format(r2))

