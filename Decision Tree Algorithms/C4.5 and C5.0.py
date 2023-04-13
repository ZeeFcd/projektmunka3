import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Generate random data
np.random.seed(42)
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a C4.5 decision tree
clf_c45 = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_c45.fit(X_train, y_train)

# Train a C5.0 decision tree
clf_c50 = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=42)
clf_c50.fit(X_train, y_train)

# Evaluate models on test data
acc_c45 = clf_c45.score(X_test, y_test)
acc_c50 = clf_c50.score(X_test, y_test)

print(f"C4.5 accuracy: {acc_c45:.2f}")
print(f"C5.0 accuracy: {acc_c50:.2f}")

