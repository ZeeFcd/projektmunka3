import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Generate random data for classification
X_classification = np.random.rand(100, 3) * 10
y_classification = np.random.randint(0, 2, 100)

# Generate random data for regression
X_regression = np.random.rand(100, 3) * 10
y_regression = np.random.rand(100) * 100

# Split data into training and testing sets for classification
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(
    X_classification, y_classification, test_size=0.2, random_state=42
)

# Split data into training and testing sets for regression
X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(
    X_regression, y_regression, test_size=0.2, random_state=42
)

# Fit and evaluate the classification tree model
classification_tree = DecisionTreeClassifier(random_state=42)
classification_tree.fit(X_train_classification, y_train_classification)
classification_tree_score = classification_tree.score(X_test_classification, y_test_classification)
print(f"Classification Tree Accuracy: {classification_tree_score:.2f}")

# Fit and evaluate the regression tree model
regression_tree = DecisionTreeRegressor(random_state=42)
regression_tree.fit(X_train_regression, y_train_regression)
regression_tree_score = regression_tree.score(X_test_regression, y_test_regression)
print(f"Regression Tree R2 Score: {regression_tree_score:.2f}")

