import numpy as np
from sklearn.linear_model import ElasticNet

np.random.seed(42)

X = np.random.randn(100, 5)

X[:,1] = X[:,0] + np.random.randn(100)*0.01

true_w = np.array([4, 4, 0, 2, 0])
y = X @ true_w + np.random.randn(100)*0.1



model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X, y)

print(f"ElasticNet model coefficients: {model.coef_}")


from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

print(f"Lasso regression model coeff: {lasso.coef_}")
