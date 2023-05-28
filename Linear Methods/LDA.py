import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Load the training data
data = pd.read_csv('training.csv')

# Split the data into features and labels
X = data.drop('ONCOGENIC', axis=1)
y = data['ONCOGENIC']

# Create a Linear Discriminant Analysis model
lda = LinearDiscriminantAnalysis()

# Train the model on the training data
lda.fit(X, y)

# Calculate the accuracy of the model
y_pred = lda.predict(X)
accuracy = accuracy_score(y, y_pred)

# Print the name of the model, the training loss, and the accuracy
print("Model: Linear Discriminant Analysis")
print("Training loss: N/A (LDA does not have a loss function)")
print("Accuracy: {:.2f}%".format(accuracy * 100))


