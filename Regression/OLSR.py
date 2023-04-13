import numpy as np
import statsmodels.api as sm

# Generate some sample data
np.random.seed(42)
X = np.random.rand(100, 5)
y = 1 + 2*X[:,0] + 3*X[:,1] + np.random.normal(size=100)

# Fit the regression model using OLS
model = sm.OLS(y, sm.add_constant(X)).fit()

# Print the summary of the model
print(model.summary())




