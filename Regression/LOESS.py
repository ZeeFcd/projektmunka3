import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# Load the dataset
data = pd.read_csv('training.csv')
# Generate random data
X = df.drop("ONCOGENIC", axis=1).values
y = df["ONCOGENIC"].values

# Apply LOESS smoothing with a span of 0.75
lowess = sm.nonparametric.lowess(y, x, frac=0.75)

# Plot the original data and the LOESS fit
fig, ax = plt.subplots()
ax.scatter(x, y, label='Original Data')
ax.plot(lowess[:, 0], lowess[:, 1], 'r-', label='LOESS Fit')
ax.legend()
plt.show()
