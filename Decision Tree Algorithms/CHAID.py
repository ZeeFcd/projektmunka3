import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate random data
np.random.seed(1234)
data = pd.DataFrame({'Age': np.random.randint(18, 65, 1000),
                     'Income': np.random.normal(50000, 10000, 1000),
                     'Sex': np.random.choice(['Male', 'Female'], 1000)})

# Define target variable
data['Target'] = np.where((data.Age >= 35) & (data.Income >= 60000), 'Yes', 'No')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Target', axis=1),
                                                    data['Target'],
                                                    test_size=0.3,
                                                    random_state=1234)

# Fit CHAID tree model
dt = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=1234)
dt.fit(X_train, y_train)

# Make predictions on testing set
y_pred = dt.predict(X_test)

# Evaluate model accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")

