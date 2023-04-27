import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Load the training data
df = pd.read_csv("training.csv")

# Split the data into input and target variables
X = df.iloc[:, :-1].values
y = df["ONCOGENIC"].values

# Create a pipeline that includes PCA and linear regression
pipeline = make_pipeline(PCA(n_components=2), LinearRegression())

# Fit the pipeline to the training data
pipeline.fit(X, y)

# Use the pipeline to make predictions on new data
new_data = [[1, 2, 3, 4]]
prediction = pipeline.predict(new_data)

print(prediction)

