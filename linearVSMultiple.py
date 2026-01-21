import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X1 = np.random.rand(100, 1) * 10          # Single feature
X2 = np.random.rand(100, 2) * 10 

y1 = 3 * X1[:, 0] + np.random.randn(100)  # Linear
y2 = 2 * X2[:, 0] + 4 * X2[:, 1] + np.random.randn(100)  # Multiple

lin_reg = LinearRegression().fit(X1, y1)
mul_reg = LinearRegression().fit(X2, y2)

# Predictions
y1_pred = lin_reg.predict(X1)
y2_pred = mul_reg.predict(X2)

# Plot 1: Linear Regression
plt.figure()
plt.scatter(X1, y1)
plt.plot(X1, y1_pred)
plt.title("Linear Regression (1 Feature)")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# Plot 2: Multiple Regression (using first feature for visualization)
plt.figure()
plt.scatter(X2[:, 0], y2)
plt.scatter(X2[:, 0], y2_pred)
plt.title("Multiple Linear Regression (2 Features)")
plt.xlabel("X1")
plt.ylabel("y")
plt.show()