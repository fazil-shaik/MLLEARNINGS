import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 0.5 * X**2 + X + np.random.randn(100, 1) * 5

# Linear Regression
linear = LinearRegression()
linear.fit(X, y)
y_linear = linear.predict(X)

# Polynomial Regression (degree = 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly = poly_reg.predict(X_poly)

# Plot Linear Regression
plt.figure()
plt.scatter(X, y)
plt.plot(X, y_linear)
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# Plot Polynomial Regression
plt.figure()
plt.scatter(X, y)
plt.plot(X, y_poly)
plt.title("Polynomial Regression (Degree 2)")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
