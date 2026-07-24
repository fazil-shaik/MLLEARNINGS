import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# Input data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])  # quadratic relationship


poly=PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly,y)

y_pred = model.predict(X_poly)


new_X = np.array([[6]])
new_X_poly = poly.transform(new_X)
print('new value is ',model.predict(new_X_poly))

plt.scatter(X,y)
plt.savefig("polynomrial11.png")
plt.plot(X, y_pred)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Regression (Degree 2)")
plt.show()
