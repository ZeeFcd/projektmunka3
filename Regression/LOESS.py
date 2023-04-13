import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# Generate random data
np.random.seed(1234)
x = np.random.uniform(low=-5, high=5, size=100)
y = np.sin(x) + np.random.normal(size=len(x))

# Sort data by x values
sorted_idx = np.argsort(x)
x_sorted = x[sorted_idx]
y_sorted = y[sorted_idx]

# Apply LOESS smoothing
fraction = 0.2 # fraction of data used for smoothing
smoothed = lowess(y_sorted, x_sorted, frac=fraction)

# Plot original data and smoothed line
plt.scatter(x, y, alpha=0.5)
plt.plot(smoothed[:, 0], smoothed[:, 1], c='orange', lw=3)
plt.title('LOESS')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
