import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

# Simple deterministic dataset (no randomness)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])  # perfectly linear: y = 2x

# Linear Regression
lr = LinearRegression()
lr.fit(X, y)
y_lr = lr.predict(X)

# Ridge Regression (with regularization)
ridge = Ridge(alpha=3)
ridge.fit(X, y)
y_ridge = ridge.predict(X)

# Plot
plt.figure()
plt.scatter(X, y)
plt.plot(X, y_lr)
plt.plot(X, y_ridge)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression vs Ridge Regression (Simple Example)")
plt.show()
