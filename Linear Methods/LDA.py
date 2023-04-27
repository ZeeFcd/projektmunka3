import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the dataset into a pandas dataframe
df = pd.read_csv("training.csv")

# Split the dataframe into predictor (X) and response (y) variables
X = df.iloc[:, :-1].values
y = df["ONCOGENIC"].values

# Create an instance of the LDA model
lda = LinearDiscriminantAnalysis()

# Fit the LDA model to the data
lda.fit(X, y)

# Use the trained LDA model to transform the data
X_lda = lda.transform(X)

