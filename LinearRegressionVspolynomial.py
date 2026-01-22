import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score

np.random.seed(42)

X = np.random.rand(100, 1) * 10
y = 3*X**2 +2*X + 1 + np.random.randn(100, 1) * 10

poly_X = np.linspace(0, 10, 100).reshape(-1, 1)
poly_Y = 3*poly_X**2 +2*poly_X + 1

linear_model = LinearRegression()
linear_model.fit(X, y)
linear_predictions = linear_model.predict(poly_X)

print("Linear predicitons are", linear_predictions[:5].flatten())

PolyModel = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2)),
    ('linear_regression', LinearRegression())
])

PolyModel.fit(X, y)
poly_predictions = PolyModel.predict(poly_X)

print("Linear regression MSE:", mean_squared_error(y, linear_model.predict(X)))
print("Polynomial regression MSE:", mean_squared_error(y, PolyModel.predict(X)))
print("Linear regression R2:", r2_score(y, linear_model.predict(X)))
print("Polynomial regression R2:", r2_score(y, PolyModel.predict(X)))

import matplotlib.pyplot as plt
plt.scatter(X, y, color='lightgray', label='Data Points')
plt.plot(poly_X, poly_Y, color='green', label='True Function', linewidth=
2)
plt.plot(poly_X, linear_predictions, color='blue', linestyle='--', label='Linear Regression')
plt.plot(poly_X, poly_predictions, color='red', linestyle='--', label='Polynomial Regression (Degree 2)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression vs Polynomial Regression')
plt.legend()
plt.show()
