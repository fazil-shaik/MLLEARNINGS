from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1,2], [2,4], [3,6]])
y = np.array([5, 10, 15])

lr = LinearRegression()
lr.fit(X, y)

print(lr.coef_)
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)  # alpha = λ
ridge.fit(X, y)

print(f"Ridge regression check coefficients {ridge.coef_}")
