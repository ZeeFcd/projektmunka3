import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# load the dataset
data = pd.read_csv('training.csv')

# separate the target variable from the features
target = data['ONCOGENIC']
features = data.drop('ONCOGENIC', axis=1)

# perform MDA
lda = LinearDiscriminantAnalysis()
lda.fit(features, target)

# print the coefficients and the explained variance ratios
print('Coefficients: ', lda.coef_)
print('Explained Variance Ratios: ', lda.explained_variance_ratio_)

