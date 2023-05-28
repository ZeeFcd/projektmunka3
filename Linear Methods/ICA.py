import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load the training dataset
data = pd.read_csv('training.csv')

# Separate the target variable from the features
X = data.drop('ONCOGENIC', axis=1)
y = data['ONCOGENIC']

# Perform ICA on the features
ica = FastICA()
X_ica = ica.fit_transform(X)

# Train a classifier on the ICA-transformed data
clf = DecisionTreeClassifier()
clf.fit(X_ica, y)

# Evaluate the model on the training data
y_pred = clf.predict(X_ica)
accuracy = accuracy_score(y, y_pred)

# Print the name of the model and the training loss
print('Independent Component Analysis (ICA)')
print('Training Loss: N/A')
print('Accuracy: %.2f' % accuracy)

