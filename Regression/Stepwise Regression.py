import numpy as np
import pandas as pd
import statsmodels.api as sm

# Generate random data
np.random.seed(123)
x1 = np.random.randn(100)
x2 = np.random.randn(100)
x3 = np.random.randn(100)
y = 2*x1 + 3*x2 + 5*x3 + np.random.randn(100)

# Create a pandas dataframe from the data
data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'y': y})

# Initialize the stepwise regression model
model = sm.WLS(data['y'], sm.add_constant(data[['x1', 'x2', 'x3']])).fit()

# Perform the stepwise regression
for i in range(3):
    pvalues = model.pvalues
    if pvalues.max() > 0.05:
        idx = pvalues.idxmax()
        model = sm.WLS(data['y'], sm.add_constant(data.drop(columns=[idx]))).fit()
    else:
        break

# Print the final model summary
print(model.summary())