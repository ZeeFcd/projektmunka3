import pandas as pd
import statsmodels.api as sm

# load training data from CSV file
df = pd.read_csv('training.csv')

# split data into dependent and independent variables
X = df.drop(['dependent_variable'], axis=1) # independent variables
y = df['dependent_variable'] # dependent variable

# add constant to independent variables
X = sm.add_constant(X)

# create OLSR model and fit it to the data
model = sm.OLS(y, X).fit()

# print the summary of the model
print(model.summary())




