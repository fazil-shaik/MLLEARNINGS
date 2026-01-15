import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(0)

X = np.linspace(-3, 3, 50).reshape(-1, 1)
y = X**2 + np.random.randn(50, 1)

# Underfitting: Linear model
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred = lin_reg.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.title("Underfitting (Linear Model)")
plt.show()
